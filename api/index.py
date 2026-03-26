from flask import Flask, request, jsonify, send_from_directory, Response, redirect, session
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import json, os, datetime, time, sys
from contextlib import nullcontext

# ── Setup ────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")
SCOPES = ["https://www.googleapis.com/auth/calendar"]

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "public"))
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-me")
llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
TZ = os.environ.get("TIMEZONE", "America/Los_Angeles")
MODEL = os.environ.get("MODEL", "gpt-4o-mini")

if os.environ.get("FLASK_DEBUG") or os.environ.get("OAUTHLIB_INSECURE_TRANSPORT"):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

DEBUG_TIMING = os.environ.get("FLASK_DEBUG") or os.environ.get("OAUTHLIB_INSECURE_TRANSPORT")

# ── Timing ──────────────────────────────────────────────────────

class RequestTimer:
    """Collects timing spans for a single request and prints a summary."""
    def __init__(self, label):
        self.label = label
        self.start = time.perf_counter()
        self.spans = []

    def span(self, name):
        return TimerSpan(name, self)

    def record(self, name, elapsed):
        self.spans.append((name, elapsed))

    def summary(self):
        if not DEBUG_TIMING:
            return
        total = time.perf_counter() - self.start
        print(f"\n\033[1;36m{'='*60}\033[0m", file=sys.stderr)
        print(f"\033[1;36m  {self.label}  —  total {total:.2f}s\033[0m", file=sys.stderr)
        print(f"\033[1;36m{'='*60}\033[0m", file=sys.stderr)
        for name, elapsed in self.spans:
            color = "\033[32m" if elapsed < 0.5 else "\033[33m" if elapsed < 2.0 else "\033[31m"
            bar = "█" * min(int(elapsed * 10), 40)
            print(f"  {color}{elapsed:6.2f}s\033[0m  {bar:40s}  {name}", file=sys.stderr)
        print(f"\033[1;36m{'─'*60}\033[0m\n", file=sys.stderr)


class TimerSpan:
    def __init__(self, name, timer):
        self.name = name
        self.timer = timer
    def __enter__(self):
        self.t0 = time.perf_counter()
        if DEBUG_TIMING:
            print(f"  \033[90m▸ {self.name} …\033[0m", file=sys.stderr)
        return self
    def __exit__(self, *_):
        elapsed = time.perf_counter() - self.t0
        self.timer.record(self.name, elapsed)
        if DEBUG_TIMING:
            color = "\033[32m" if elapsed < 0.5 else "\033[33m" if elapsed < 2.0 else "\033[31m"
            print(f"  {color}✓ {self.name}: {elapsed:.2f}s\033[0m", file=sys.stderr)
        self.elapsed = elapsed

_req_timer = None  # set per-request

SYSTEM_PROMPT = """CalBot — calendar assistant. Today: {today}. Timezone: {tz}.

Create events with create_calendar_event (ask for time if missing, default 1hr, always include a description).
Check schedule with list_upcoming_events. Delete: list first, then delete_calendar_event.
Brief acknowledgment before tool calls. **Bold** names/dates/times. Short paragraphs, no bullet points. Use [links](url)."""

TOOLS = [
    {"type": "function", "function": {
        "name": "create_calendar_event",
        "description": "Create a new event on Google Calendar.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "Event title"},
            "start_datetime": {"type": "string", "description": "ISO 8601 start"},
            "end_datetime": {"type": "string", "description": "ISO 8601 end"},
            "description": {"type": "string", "description": "1-2 sentence event description"},
            "location": {"type": "string", "description": "Location or empty string"},
        }, "required": ["summary", "start_datetime", "end_datetime", "description", "location"],
           "additionalProperties": False}, "strict": True}},
    {"type": "function", "function": {
        "name": "list_upcoming_events",
        "description": "List upcoming events with event_id for deletion.",
        "parameters": {"type": "object", "properties": {
            "days_ahead": {"type": "integer", "description": "Days to look ahead"},
        }, "required": ["days_ahead"], "additionalProperties": False}, "strict": True}},
    {"type": "function", "function": {
        "name": "delete_calendar_event",
        "description": "Delete a calendar event by event_id.",
        "parameters": {"type": "object", "properties": {
            "event_id": {"type": "string", "description": "Google Calendar event ID"},
            "summary": {"type": "string", "description": "Event title for display"},
        }, "required": ["event_id", "summary"], "additionalProperties": False}, "strict": True}},
]


# ── SSE helpers ──────────────────────────────────────────────────

def sse(typ, data):
    return f"data: {json.dumps({'type': typ, **data})}\n\n"

def sse_response(gen):
    return Response(gen, mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

def sys_msg():
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    return {"role": "system", "content": SYSTEM_PROMPT.format(today=today, tz=TZ)}

def stream_reply(messages, history, label="stream_reply"):
    """Simple LLM stream → SSE tokens → done."""
    global _req_timer
    t0 = time.perf_counter()
    ttft = None
    stream = llm.chat.completions.create(model=MODEL, messages=messages, stream=True)
    content = ""
    for chunk in stream:
        d = chunk.choices[0].delta
        if d.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
                if _req_timer:
                    _req_timer.record(f"{label} TTFT", ttft)
            content += d.content
            yield sse("token", {"content": d.content})
    total = time.perf_counter() - t0
    if _req_timer:
        _req_timer.record(f"{label} total", total)
    history.append({"role": "assistant", "content": content})
    yield sse("done", {"history": history})

def stream_with_tools(messages, label="LLM call"):
    """Stream LLM, yield tokens and collect tool calls."""
    global _req_timer
    t0 = time.perf_counter()
    ttft = None
    stream = llm.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, stream=True)
    content, tcs = "", {}
    for chunk in stream:
        d = chunk.choices[0].delta
        if d.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
                if _req_timer:
                    _req_timer.record(f"{label} TTFT", ttft)
            content += d.content
            yield "tok", d.content
        if d.tool_calls:
            if ttft is None:
                ttft = time.perf_counter() - t0
                if _req_timer:
                    _req_timer.record(f"{label} TTFT (tool)", ttft)
            for tc in d.tool_calls:
                i = tc.index
                if i not in tcs:
                    tcs[i] = {"id": "", "name": "", "arguments": ""}
                if tc.id: tcs[i]["id"] = tc.id
                if tc.function:
                    if tc.function.name: tcs[i]["name"] += tc.function.name
                    if tc.function.arguments: tcs[i]["arguments"] += tc.function.arguments
    total = time.perf_counter() - t0
    if _req_timer:
        _req_timer.record(f"{label} total", total)
    yield "done", (content, tcs)


def _emit_group(items, single_type, batch_type, history):
    if len(items) == 1:
        tc = items[0]
        yield sse(single_type, {"event": json.loads(tc["arguments"]),
                  "tool_call_id": tc["id"], "tool_call": tc, "history": history})
    elif len(items) > 1:
        yield sse(batch_type, {"items": [{"event": json.loads(tc["arguments"]),
                  "tool_call_id": tc["id"], "tool_call": tc} for tc in items], "history": history})


def process_tool_calls(tcs, content, history, token):
    """Group tool calls, emit confirmations or auto-execute list queries."""
    groups = {"create_calendar_event": [], "delete_calendar_event": [], "list_upcoming_events": []}
    for i in sorted(tcs):
        tc = tcs[i]
        if tc["name"] in groups:
            groups[tc["name"]].append(tc)

    all_tcs = groups["create_calendar_event"] + groups["delete_calendar_event"] + groups["list_upcoming_events"]
    hist_entry = {"role": "assistant", "content": content, "tool_calls": [
        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
        for tc in all_tcs]}
    updated = history + [hist_entry]

    yield from _emit_group(groups["create_calendar_event"], "confirm", "confirm_batch", updated)
    yield from _emit_group(groups["delete_calendar_event"], "confirm_delete", "confirm_delete_batch", updated)

    for tc in groups["list_upcoming_events"]:
        yield sse("status", {"message": "Checking your calendar"})
        with _req_timer.span("list_upcoming_events (Google API)") if _req_timer else nullcontext():
            result = list_upcoming_events(**json.loads(tc["arguments"]), _token=token)
        updated.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
        yield sse("status", {"message": "Preparing response"})
        yield sse("tool_used", {"name": "list_upcoming_events"})

        follow_content, follow_tcs, streamed_tokens = "", {}, []
        for typ, payload in stream_with_tools([sys_msg()] + updated[-20:], label="follow-up LLM"):
            if typ == "tok":
                follow_content += payload
                yield sse("token", {"content": payload})
                streamed_tokens.append(payload)
            else:
                _, follow_tcs = payload

        if follow_tcs:
            # Follow-up wants to call tools (e.g. delete after listing)
            follow_all = list(follow_tcs.values())
            hist_entry2 = {"role": "assistant", "content": follow_content, "tool_calls": [
                {"id": ft["id"], "type": "function",
                 "function": {"name": ft["name"], "arguments": ft["arguments"]}}
                for ft in follow_all]}
            updated.append(hist_entry2)

            f_creates = [ft for ft in follow_all if ft["name"] == "create_calendar_event"]
            f_deletes = [ft for ft in follow_all if ft["name"] == "delete_calendar_event"]
            yield from _emit_group(f_creates, "confirm", "confirm_batch", updated)
            yield from _emit_group(f_deletes, "confirm_delete", "confirm_delete_batch", updated)
        else:
            # Tokens already streamed above — just finalize
            updated.append({"role": "assistant", "content": follow_content})
            yield sse("done", {"history": updated})


# ── Auth helpers ─────────────────────────────────────────────────

def _oauth_config():
    creds_file = os.path.join(BASE_DIR, "credentials.json")
    if os.path.exists(creds_file):
        with open(creds_file) as f:
            return json.load(f)
    cid, csec = os.environ.get("GOOGLE_CLIENT_ID"), os.environ.get("GOOGLE_CLIENT_SECRET")
    if cid and csec:
        return {"web": {"client_id": cid, "client_secret": csec,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"}}
    return None

def _creds(data):
    return Credentials(token=data["token"], refresh_token=data["refresh_token"],
        token_uri=data["token_uri"], client_id=data["client_id"],
        client_secret=data["client_secret"], scopes=SCOPES)

def _token():
    """Get token string. Must call in request context."""
    t = session.get("google_token")
    if t: return t
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH) as f: return f.read()
    return os.environ.get("GOOGLE_TOKEN")

def _redirect_uri():
    cfg = _oauth_config()
    if cfg:
        key = "web" if "web" in cfg else "installed"
        uris = cfg.get(key, {}).get("redirect_uris", [])
        if uris: return uris[0]
    return request.url_root.rstrip("/") + "/api/auth/callback"


# ── Calendar ─────────────────────────────────────────────────────

_cal_cache = {}  # keyed by token hash → (service, creds)

def _cal(token_str=None):
    global _req_timer
    t = token_str or _token()
    if not t: raise Exception("Not authenticated. Please reconnect Google Calendar.")

    cache_key = hash(t)
    if cache_key in _cal_cache:
        service, creds = _cal_cache[cache_key]
        if not creds.expired:
            if _req_timer:
                _req_timer.record("_cal (cached)", 0.0)
            return service

    t0 = time.perf_counter()
    creds = _creds(json.loads(t))
    if creds.expired and creds.refresh_token:
        rt0 = time.perf_counter()
        creds.refresh(Request())
        if _req_timer:
            _req_timer.record("creds.refresh", time.perf_counter() - rt0)

    bt0 = time.perf_counter()
    service = build("calendar", "v3", credentials=creds, cache_discovery=True)
    if _req_timer:
        _req_timer.record("discovery.build", time.perf_counter() - bt0)

    _cal_cache[cache_key] = (service, creds)
    total = time.perf_counter() - t0
    if _req_timer:
        _req_timer.record("_cal total (built)", total)
    return service

def create_calendar_event(summary, start_datetime, end_datetime, description="", location="", _token=None):
    event = {"summary": summary, "start": {"dateTime": start_datetime, "timeZone": TZ},
             "end": {"dateTime": end_datetime, "timeZone": TZ}}
    if description: event["description"] = description
    if location: event["location"] = location
    r = _cal(_token).events().insert(calendarId="primary", body=event).execute()
    return {"status": "created", "summary": r["summary"], "start": r["start"].get("dateTime"),
            "end": r["end"].get("dateTime"), "link": r.get("htmlLink", "")}

def list_upcoming_events(days_ahead=7, _token=None):
    now = datetime.datetime.utcnow()
    r = _cal(_token).events().list(calendarId="primary", timeMin=now.isoformat()+"Z",
        timeMax=(now + datetime.timedelta(days=days_ahead)).isoformat()+"Z",
        maxResults=20, singleEvents=True, orderBy="startTime").execute()
    return {"events": [{"event_id": e["id"], "summary": e.get("summary", "No title"),
        "start": e["start"].get("dateTime", e["start"].get("date")),
        "end": e["end"].get("dateTime", e["end"].get("date")),
        "location": e.get("location", "")} for e in r.get("items", [])]}

def delete_calendar_event(event_id, summary="", _token=None):
    try:
        _cal(_token).events().delete(calendarId="primary", eventId=event_id).execute()
        return {"status": "deleted", "event_id": event_id, "summary": summary}
    except Exception as e:
        return {"status": "error", "event_id": event_id, "summary": summary, "error": str(e)}

FUNCS = {"create_calendar_event": create_calendar_event,
         "list_upcoming_events": list_upcoming_events,
         "delete_calendar_event": delete_calendar_event}


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/auth/status")
def auth_status():
    for src in [lambda: session.get("google_token"),
                lambda: open(TOKEN_PATH).read() if os.path.exists(TOKEN_PATH) else None,
                lambda: os.environ.get("GOOGLE_TOKEN")]:
        try:
            t = src()
            if not t: continue
            c = _creds(json.loads(t))
            if c.valid: return jsonify({"authenticated": True})
            if c.expired and c.refresh_token:
                c.refresh(Request()); session["google_token"] = c.to_json()
                return jsonify({"authenticated": True})
        except Exception: pass
    return jsonify({"authenticated": False})

@app.route("/api/auth/login")
def auth_login():
    from google_auth_oauthlib.flow import Flow
    cfg = _oauth_config()
    if not cfg: return jsonify({"error": "No OAuth credentials configured."}), 500
    flow = Flow.from_client_config(cfg, scopes=SCOPES, redirect_uri=_redirect_uri())
    url, state = flow.authorization_url(access_type="offline", prompt="consent")
    session["oauth_state"], session["code_verifier"] = state, flow.code_verifier
    return redirect(url)

@app.route("/api/auth/callback")
def auth_callback():
    from google_auth_oauthlib.flow import Flow
    cfg = _oauth_config()
    if not cfg: return jsonify({"error": "No OAuth credentials configured."}), 500
    uri = _redirect_uri()
    flow = Flow.from_client_config(cfg, scopes=SCOPES, redirect_uri=uri)
    flow.code_verifier = session.pop("code_verifier", None)
    resp = request.url.replace("://127.0.0.1:", "://localhost:") if "localhost" in uri else request.url.replace("://localhost:", "://127.0.0.1:")
    flow.fetch_token(authorization_response=resp)
    session["google_token"] = flow.credentials.to_json()
    try:
        with open(TOKEN_PATH, "w") as f: f.write(flow.credentials.to_json())
    except OSError: pass
    return redirect("/")

@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("google_token", None)
    try:
        if os.path.exists(TOKEN_PATH): os.remove(TOKEN_PATH)
    except OSError: pass
    return jsonify({"status": "ok"})

@app.route("/api/chat", methods=["POST"])
def chat():
    global _req_timer
    _req_timer = RequestTimer("POST /api/chat")
    data = request.json
    msg = data.get("message", "").strip()
    history = data.get("history", [])
    if not msg: return jsonify({"error": "empty"}), 400
    history.append({"role": "user", "content": msg})
    messages = [sys_msg()] + history[-20:]
    token = _token()
    timer = _req_timer
    def generate():
        global _req_timer
        _req_timer = timer
        try:
            content, tcs = "", {}
            for typ, payload in stream_with_tools(messages, label="primary LLM"):
                if typ == "tok":
                    content += payload
                    yield sse("token", {"content": payload})
                else:
                    content, tcs = payload
            if tcs:
                yield from process_tool_calls(tcs, content, history, token)
            else:
                history.append({"role": "assistant", "content": content})
                yield sse("done", {"history": history})
        except Exception as e:
            yield sse("token", {"content": f"Error: {e}"})
            yield sse("done", {"history": history})
        finally:
            timer.summary()
    return sse_response(generate())

@app.route("/api/confirm", methods=["POST"])
def confirm_event():
    global _req_timer
    _req_timer = RequestTimer("POST /api/confirm")
    data = request.json
    history, tc = data["history"], data["tool_call"]
    token = _token()
    timer = _req_timer
    status_label = {"create_calendar_event": "Creating event", "delete_calendar_event": "Deleting event"}.get(tc["name"], "Working")
    def generate():
        global _req_timer
        _req_timer = timer
        yield sse("status", {"message": status_label})
        with timer.span(f"{tc['name']} (Google API)"):
            result = FUNCS[tc["name"]](**json.loads(tc["arguments"]), _token=token)
        history.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
        yield sse("tool_used", {"name": tc["name"]})
        yield from stream_reply([sys_msg()] + history[-20:], history, label="confirm reply")
        timer.summary()
    return sse_response(generate())

@app.route("/api/confirm_batch", methods=["POST"])
def confirm_batch():
    global _req_timer
    _req_timer = RequestTimer("POST /api/confirm_batch")
    data = request.json
    history = data["history"]
    token = _token()
    timer = _req_timer
    tool_calls = data.get("tool_calls", [])
    rejected = data.get("rejected_tool_calls", [])
    def generate():
        global _req_timer
        _req_timer = timer
        for i, tc in enumerate(tool_calls):
            label = {"create_calendar_event": "Creating event", "delete_calendar_event": "Deleting event"}.get(tc["name"], "Working")
            yield sse("status", {"message": f"{label} ({i+1}/{len(tool_calls)})" if len(tool_calls) > 1 else label})
            with timer.span(f"{tc['name']} (Google API)"):
                r = FUNCS[tc["name"]](**json.loads(tc["arguments"]), _token=token)
            history.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(r)})
        for tc in rejected:
            history.append({"role": "tool", "tool_call_id": tc["id"],
                            "content": json.dumps({"status": "cancelled", "reason": "User declined."})})
        yield sse("tool_used", {"name": tool_calls[0]["name"] if tool_calls else "batch"})
        yield from stream_reply([sys_msg()] + history[-20:], history, label="batch reply")
        timer.summary()
    return sse_response(generate())

@app.route("/api/reject", methods=["POST"])
def reject_event():
    global _req_timer
    _req_timer = RequestTimer("POST /api/reject")
    data = request.json
    history = data["history"]
    history.append({"role": "tool", "tool_call_id": data["tool_call"]["id"],
                    "content": json.dumps({"status": "cancelled", "reason": "User declined."})})
    timer = _req_timer
    def generate():
        global _req_timer
        _req_timer = timer
        yield from stream_reply([sys_msg()] + history[-20:], history, label="reject reply")
        timer.summary()
    return sse_response(generate())

@app.route("/api/reject_batch", methods=["POST"])
def reject_batch():
    global _req_timer
    _req_timer = RequestTimer("POST /api/reject_batch")
    data = request.json
    history = data["history"]
    for tc in data.get("tool_calls", []):
        history.append({"role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps({"status": "cancelled", "reason": "User declined."})})
    timer = _req_timer
    def generate():
        global _req_timer
        _req_timer = timer
        yield from stream_reply([sys_msg()] + history[-20:], history, label="reject_batch reply")
        timer.summary()
    return sse_response(generate())
