from flask import Flask, request, jsonify, send_from_directory, Response, redirect, session
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import json
import os
import datetime

# ── App setup ────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public")
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "..", "token.json")
SCOPES = ["https://www.googleapis.com/auth/calendar"]

app = Flask(__name__, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-me")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
TIMEZONE = os.environ.get("TIMEZONE", "America/Los_Angeles")

if os.environ.get("FLASK_DEBUG") or os.environ.get("OAUTHLIB_INSECURE_TRANSPORT"):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

SYSTEM_PROMPT = """You are CalBot, a sharp and friendly calendar assistant. You help the user manage their Google Calendar.

Today is {today}. The user's timezone is {tz}.

Rules:
- When the user wants to create an event, extract ALL details and call create_calendar_event.
- If they don't give a time, ask for it. Default duration is 1 hour.
- ALWAYS include a useful description for events. Summarize the purpose of the event in 1-2 sentences.
- When they ask about their schedule, call list_upcoming_events.
- Be concise. After creating events, confirm with the details.
- When the user wants to delete an event, first call list_upcoming_events to find it, then call delete_calendar_event with the event_id.
- Before calling any tool, always include a brief acknowledgment message (1 sentence max). For example: "Sure, let me set that up!" or "Checking your schedule now."
- When sharing links, ALWAYS use markdown format: [descriptive text](url). Never paste raw URLs."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new event on Google Calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start_datetime": {"type": "string", "description": "ISO 8601 start, e.g. 2026-03-01T10:00:00"},
                    "end_datetime": {"type": "string", "description": "ISO 8601 end, e.g. 2026-03-01T11:00:00"},
                    "description": {"type": "string", "description": "A useful 1-2 sentence description of the event's purpose"},
                    "location": {"type": "string", "description": "Location, empty string if none"},
                },
                "required": ["summary", "start_datetime", "end_datetime", "description", "location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_upcoming_events",
            "description": "List upcoming events from Google Calendar. Returns event_id for each event which can be used to delete events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_ahead": {"type": "integer", "description": "Number of days to look ahead"},
                },
                "required": ["days_ahead"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete an event from Google Calendar by its event_id. Always confirm with the user before calling this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The Google Calendar event ID"},
                    "summary": {"type": "string", "description": "The event title (for confirmation display)"},
                },
                "required": ["event_id", "summary"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


# ── Helpers ──────────────────────────────────────────────────────

def sse(event_type, data):
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


def streaming_response(generator):
    return Response(generator, mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


def build_system_message():
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    return {"role": "system", "content": SYSTEM_PROMPT.format(today=today, tz=TIMEZONE)}


def stream_llm(messages, history):
    """Stream an LLM response and yield SSE events. Returns via done event."""
    stream = client.chat.completions.create(
        model="gpt-5-nano", messages=messages, tools=TOOLS, stream=True,
    )
    content = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            yield sse("token", {"content": delta.content})
    history.append({"role": "assistant", "content": content})
    yield sse("done", {"history": history})


def collect_stream(messages):
    """Stream an LLM response, collecting content and tool calls."""
    stream = client.chat.completions.create(
        model="gpt-5-nano", messages=messages, tools=TOOLS,
        tool_choice="auto", stream=True,
    )
    content = ""
    tool_calls = {}
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            yield "token", delta.content
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments
    yield "result", (content, tool_calls)


def emit_confirmations(tool_calls, content, history, token_str):
    """Group tool calls by type and emit appropriate SSE events."""
    creates, deletes, lists = [], [], []
    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        if tc["name"] == "create_calendar_event":
            creates.append(tc)
        elif tc["name"] == "delete_calendar_event":
            deletes.append(tc)
        elif tc["name"] == "list_upcoming_events":
            lists.append(tc)

    all_tcs = creates + deletes + lists
    tc_history_entry = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in all_tcs
        ],
    }
    updated_history = history + [tc_history_entry]

    # Creates
    if len(creates) == 1:
        tc = creates[0]
        yield sse("confirm", {
            "event": json.loads(tc["arguments"]),
            "tool_call_id": tc["id"], "tool_call": tc,
            "history": updated_history,
        })
    elif len(creates) > 1:
        items = [{"event": json.loads(tc["arguments"]),
                  "tool_call_id": tc["id"], "tool_call": tc} for tc in creates]
        yield sse("confirm_batch", {"items": items, "history": updated_history})

    # Deletes
    if len(deletes) == 1:
        tc = deletes[0]
        yield sse("confirm_delete", {
            "event": json.loads(tc["arguments"]),
            "tool_call_id": tc["id"], "tool_call": tc,
            "history": updated_history,
        })
    elif len(deletes) > 1:
        items = [{"event": json.loads(tc["arguments"]),
                  "tool_call_id": tc["id"], "tool_call": tc} for tc in deletes]
        yield sse("confirm_delete_batch", {"items": items, "history": updated_history})

    # List events (auto-execute, then stream follow-up)
    for tc in lists:
        args = json.loads(tc["arguments"])
        result = list_upcoming_events(**args, _token=token_str)
        updated_history.append({
            "role": "tool", "tool_call_id": tc["id"],
            "content": json.dumps(result),
        })
        yield sse("tool_used", {"name": "list_upcoming_events"})

        follow_messages = [build_system_message()] + updated_history[-20:]
        follow_content = ""
        follow_tc = {}
        for event_type, payload in collect_stream(follow_messages):
            if event_type == "token":
                follow_content = follow_content  # already yielded by collect_stream
                yield sse("token", {"content": payload})
            elif event_type == "result":
                follow_content, follow_tc = payload

        if follow_tc:
            yield from emit_confirmations(follow_tc, follow_content, updated_history, token_str)
        else:
            updated_history.append({"role": "assistant", "content": follow_content})
            yield sse("done", {"history": updated_history})


# ── Auth helpers ─────────────────────────────────────────────────

def _get_oauth_client_config():
    creds_file = os.path.join(os.path.dirname(__file__), "..", "credentials.json")
    if os.path.exists(creds_file):
        with open(creds_file) as f:
            return json.load(f)
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if client_id and client_secret:
        return {"web": {
            "client_id": client_id, "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }}
    return None


def _load_creds(token_data):
    return Credentials(
        token=token_data["token"], refresh_token=token_data["refresh_token"],
        token_uri=token_data["token_uri"], client_id=token_data["client_id"],
        client_secret=token_data["client_secret"], scopes=SCOPES,
    )


def _get_token_str():
    """Get token string from session, file, or env. Must be called in request context."""
    token_str = session.get("google_token")
    if token_str:
        return token_str
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH) as f:
            return f.read()
    return os.environ.get("GOOGLE_TOKEN")


def _get_redirect_uri():
    client_config = _get_oauth_client_config()
    if client_config:
        key = "web" if "web" in client_config else "installed"
        uris = client_config.get(key, {}).get("redirect_uris", [])
        if uris:
            return uris[0]
    return request.url_root.rstrip("/") + "/api/auth/callback"


# ── Calendar helpers ─────────────────────────────────────────────

def get_calendar_service(token_str=None):
    from googleapiclient.discovery import build
    if not token_str:
        token_str = _get_token_str()
    if not token_str:
        raise Exception("No Google Calendar credentials found. Please reconnect.")
    creds = _load_creds(json.loads(token_str))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("calendar", "v3", credentials=creds)


def create_calendar_event(summary, start_datetime, end_datetime, description="", location="", _token=None):
    service = get_calendar_service(token_str=_token)
    event = {"summary": summary,
             "start": {"dateTime": start_datetime, "timeZone": TIMEZONE},
             "end": {"dateTime": end_datetime, "timeZone": TIMEZONE}}
    if description:
        event["description"] = description
    if location:
        event["location"] = location
    created = service.events().insert(calendarId="primary", body=event).execute()
    return {"status": "created", "summary": created["summary"],
            "start": created["start"].get("dateTime"),
            "end": created["end"].get("dateTime"),
            "link": created.get("htmlLink", "")}


def list_upcoming_events(days_ahead=7, _token=None):
    service = get_calendar_service(token_str=_token)
    now = datetime.datetime.utcnow()
    time_min = now.isoformat() + "Z"
    time_max = (now + datetime.timedelta(days=days_ahead)).isoformat() + "Z"
    result = service.events().list(
        calendarId="primary", timeMin=time_min, timeMax=time_max,
        maxResults=20, singleEvents=True, orderBy="startTime",
    ).execute()
    return {"events": [
        {"event_id": e["id"], "summary": e.get("summary", "No title"),
         "start": e["start"].get("dateTime", e["start"].get("date")),
         "end": e["end"].get("dateTime", e["end"].get("date")),
         "location": e.get("location", "")}
        for e in result.get("items", [])
    ]}


def delete_calendar_event(event_id, summary="", _token=None):
    service = get_calendar_service(token_str=_token)
    try:
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return {"status": "deleted", "event_id": event_id, "summary": summary}
    except Exception as e:
        return {"status": "error", "event_id": event_id, "summary": summary, "error": str(e)}


FUNC_MAP = {
    "create_calendar_event": create_calendar_event,
    "list_upcoming_events": list_upcoming_events,
    "delete_calendar_event": delete_calendar_event,
}


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/auth/status")
def auth_status():
    for source in [
        lambda: session.get("google_token"),
        lambda: open(TOKEN_PATH).read() if os.path.exists(TOKEN_PATH) else None,
        lambda: os.environ.get("GOOGLE_TOKEN"),
    ]:
        try:
            token_str = source()
            if not token_str:
                continue
            creds = _load_creds(json.loads(token_str))
            if creds.valid:
                return jsonify({"authenticated": True})
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                session["google_token"] = creds.to_json()
                return jsonify({"authenticated": True})
        except Exception:
            pass
    return jsonify({"authenticated": False})


@app.route("/api/auth/login")
def auth_login():
    from google_auth_oauthlib.flow import Flow
    client_config = _get_oauth_client_config()
    if not client_config:
        return jsonify({"error": "No OAuth credentials configured."}), 500
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=_get_redirect_uri())
    auth_url, state = flow.authorization_url(access_type="offline", prompt="consent")
    session["oauth_state"] = state
    session["code_verifier"] = flow.code_verifier
    return redirect(auth_url)


@app.route("/api/auth/callback")
def auth_callback():
    from google_auth_oauthlib.flow import Flow
    client_config = _get_oauth_client_config()
    if not client_config:
        return jsonify({"error": "No OAuth credentials configured."}), 500
    redirect_uri = _get_redirect_uri()
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=redirect_uri)
    flow.code_verifier = session.pop("code_verifier", None)
    auth_response = request.url
    if "127.0.0.1" in redirect_uri:
        auth_response = auth_response.replace("://localhost:", "://127.0.0.1:")
    else:
        auth_response = auth_response.replace("://127.0.0.1:", "://localhost:")
    flow.fetch_token(authorization_response=auth_response)
    session["google_token"] = flow.credentials.to_json()
    try:
        with open(TOKEN_PATH, "w") as f:
            f.write(flow.credentials.to_json())
    except OSError:
        pass
    return redirect("/")


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("google_token", None)
    try:
        if os.path.exists(TOKEN_PATH):
            os.remove(TOKEN_PATH)
    except OSError:
        pass
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "").strip()
    history = data.get("history", [])
    if not user_msg:
        return jsonify({"error": "empty"}), 400

    history.append({"role": "user", "content": user_msg})
    sys_msg = build_system_message()
    messages = [sys_msg] + history[-20:]
    token_str = _get_token_str()

    def generate():
        try:
            content = ""
            tool_calls = {}
            for event_type, payload in collect_stream(messages):
                if event_type == "token":
                    content += payload
                    yield sse("token", {"content": payload})
                elif event_type == "result":
                    content, tool_calls = payload

            if tool_calls:
                yield from emit_confirmations(tool_calls, content, history, token_str)
            else:
                history.append({"role": "assistant", "content": content})
                yield sse("done", {"history": history})
        except Exception as e:
            yield sse("token", {"content": f"Error: {str(e)}"})
            yield sse("done", {"history": history})

    return streaming_response(generate())


@app.route("/api/confirm", methods=["POST"])
def confirm_event():
    data = request.json
    history = data.get("history", [])
    tool_call = data.get("tool_call")
    token_str = _get_token_str()

    func = FUNC_MAP[tool_call["name"]]
    result = func(**json.loads(tool_call["arguments"]), _token=token_str)
    history.append({"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result)})

    messages = [build_system_message()] + history[-20:]

    def generate():
        yield sse("tool_used", {"name": tool_call["name"]})
        yield from stream_llm(messages, history)

    return streaming_response(generate())


@app.route("/api/confirm_batch", methods=["POST"])
def confirm_batch():
    data = request.json
    history = data.get("history", [])
    accepted = data.get("tool_calls", [])
    rejected = data.get("rejected_tool_calls", [])
    token_str = _get_token_str()

    for tc in accepted:
        result = FUNC_MAP[tc["name"]](**json.loads(tc["arguments"]), _token=token_str)
        history.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
    for tc in rejected:
        history.append({"role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps({"status": "cancelled", "reason": "User declined."})})

    messages = [build_system_message()] + history[-20:]

    def generate():
        yield sse("tool_used", {"name": accepted[0]["name"] if accepted else "batch"})
        yield from stream_llm(messages, history)

    return streaming_response(generate())


@app.route("/api/reject", methods=["POST"])
def reject_event():
    data = request.json
    history = data.get("history", [])
    tool_call = data.get("tool_call")
    history.append({"role": "tool", "tool_call_id": tool_call["id"],
                    "content": json.dumps({"status": "cancelled", "reason": "User declined."})})
    messages = [build_system_message()] + history[-20:]

    def generate():
        yield from stream_llm(messages, history)

    return streaming_response(generate())


@app.route("/api/reject_batch", methods=["POST"])
def reject_batch():
    data = request.json
    history = data.get("history", [])
    for tc in data.get("tool_calls", []):
        history.append({"role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps({"status": "cancelled", "reason": "User declined."})})
    messages = [build_system_message()] + history[-20:]

    def generate():
        yield from stream_llm(messages, history)

    return streaming_response(generate())
