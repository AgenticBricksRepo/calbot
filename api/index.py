from flask import Flask, request, jsonify, send_from_directory, Response, redirect, session
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import json, os, datetime

# ── Setup ────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")
SCOPES = ["https://www.googleapis.com/auth/calendar"]

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "public"))
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-me")
llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
TZ = os.environ.get("TIMEZONE", "America/Los_Angeles")
MODEL = "gpt-5-nano"

if os.environ.get("FLASK_DEBUG") or os.environ.get("OAUTHLIB_INSECURE_TRANSPORT"):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

SYSTEM_PROMPT = """You are CalBot, a sharp and friendly calendar assistant. You help the user manage their Google Calendar.

Today is {today}. The user's timezone is {tz}.

Rules:
- When the user wants to create an event, extract ALL details and call create_calendar_event.
- If they don't give a time, ask for it. Default duration is 1 hour.
- ALWAYS include a useful description for events (1-2 sentences).
- When they ask about their schedule, call list_upcoming_events.
- Be concise. After creating events, confirm with the details.
- To delete an event, first call list_upcoming_events to find it, then call delete_calendar_event with the event_id.
- Before calling any tool, include a brief acknowledgment (1 sentence max).
- Use markdown formatting. **Bold** important info like event names, dates, and times. Do NOT use bullet points or lists — write in short paragraphs or single lines. Use [links](url) for calendar links. Never paste raw URLs."""

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

def stream_reply(messages, history):
    """Simple LLM stream → SSE tokens → done."""
    stream = llm.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, stream=True)
    content = ""
    for chunk in stream:
        d = chunk.choices[0].delta
        if d.content:
            content += d.content
            yield sse("token", {"content": d.content})
    history.append({"role": "assistant", "content": content})
    yield sse("done", {"history": history})

def stream_with_tools(messages):
    """Stream LLM, yield tokens and collect tool calls."""
    stream = llm.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", stream=True)
    content, tcs = "", {}
    for chunk in stream:
        d = chunk.choices[0].delta
        if d.content:
            content += d.content
            yield "tok", d.content
        if d.tool_calls:
            for tc in d.tool_calls:
                i = tc.index
                if i not in tcs:
                    tcs[i] = {"id": "", "name": "", "arguments": ""}
                if tc.id: tcs[i]["id"] = tc.id
                if tc.function:
                    if tc.function.name: tcs[i]["name"] += tc.function.name
                    if tc.function.arguments: tcs[i]["arguments"] += tc.function.arguments
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
        result = list_upcoming_events(**json.loads(tc["arguments"]), _token=token)
        updated.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
        yield sse("tool_used", {"name": "list_upcoming_events"})

        follow_content, follow_tcs, buffered = "", {}, []
        for typ, payload in stream_with_tools([sys_msg()] + updated[-20:]):
            if typ == "tok":
                follow_content += payload
                buffered.append(payload)
            else:
                _, follow_tcs = payload

        if follow_tcs:
            # Follow-up wants to call tools (e.g. delete after listing)
            # Don't stream the follow-up text — just put it in history and show cards
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
            # No tool calls — stream all the buffered text
            for t in buffered:
                yield sse("token", {"content": t})
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

def _cal(token_str=None):
    from googleapiclient.discovery import build
    t = token_str or _token()
    if not t: raise Exception("Not authenticated. Please reconnect Google Calendar.")
    creds = _creds(json.loads(t))
    if creds.expired and creds.refresh_token: creds.refresh(Request())
    return build("calendar", "v3", credentials=creds)

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
    data = request.json
    msg = data.get("message", "").strip()
    history = data.get("history", [])
    if not msg: return jsonify({"error": "empty"}), 400
    history.append({"role": "user", "content": msg})
    messages = [sys_msg()] + history[-20:]
    token = _token()
    def generate():
        try:
            content, tcs = "", {}
            for typ, payload in stream_with_tools(messages):
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
    return sse_response(generate())

@app.route("/api/confirm", methods=["POST"])
def confirm_event():
    data = request.json
    history, tc = data["history"], data["tool_call"]
    token = _token()
    result = FUNCS[tc["name"]](**json.loads(tc["arguments"]), _token=token)
    history.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
    def generate():
        yield sse("tool_used", {"name": tc["name"]})
        yield from stream_reply([sys_msg()] + history[-20:], history)
    return sse_response(generate())

@app.route("/api/confirm_batch", methods=["POST"])
def confirm_batch():
    data = request.json
    history = data["history"]
    token = _token()
    for tc in data.get("tool_calls", []):
        r = FUNCS[tc["name"]](**json.loads(tc["arguments"]), _token=token)
        history.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(r)})
    for tc in data.get("rejected_tool_calls", []):
        history.append({"role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps({"status": "cancelled", "reason": "User declined."})})
    def generate():
        yield sse("tool_used", {"name": data["tool_calls"][0]["name"] if data.get("tool_calls") else "batch"})
        yield from stream_reply([sys_msg()] + history[-20:], history)
    return sse_response(generate())

@app.route("/api/reject", methods=["POST"])
def reject_event():
    data = request.json
    history = data["history"]
    history.append({"role": "tool", "tool_call_id": data["tool_call"]["id"],
                    "content": json.dumps({"status": "cancelled", "reason": "User declined."})})
    def generate():
        yield from stream_reply([sys_msg()] + history[-20:], history)
    return sse_response(generate())

@app.route("/api/reject_batch", methods=["POST"])
def reject_batch():
    data = request.json
    history = data["history"]
    for tc in data.get("tool_calls", []):
        history.append({"role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps({"status": "cancelled", "reason": "User declined."})})
    def generate():
        yield from stream_reply([sys_msg()] + history[-20:], history)
    return sse_response(generate())
