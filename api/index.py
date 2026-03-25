from flask import Flask, request, jsonify, send_from_directory, Response, redirect, session, stream_with_context
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import json
import os
import datetime

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public")
app = Flask(__name__, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-me")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── Config ────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
TIMEZONE = os.environ.get("TIMEZONE", "America/Los_Angeles")
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "..", "token.json")


# Allow OAuth over HTTP for local development
if os.environ.get("FLASK_DEBUG") or os.environ.get("OAUTHLIB_INSECURE_TRANSPORT"):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"


# ── Auth endpoints ───────────────────────────────────────────────

def _get_oauth_client_config():
    """Load OAuth client config from credentials.json or env vars."""
    creds_file = os.path.join(os.path.dirname(__file__), "..", "credentials.json")
    if os.path.exists(creds_file):
        with open(creds_file) as f:
            return json.load(f)

    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if client_id and client_secret:
        return {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }
    return None


def _load_creds_from_token_data(token_data):
    """Build Credentials from a token dict."""
    return Credentials(
        token=token_data["token"],
        refresh_token=token_data["refresh_token"],
        token_uri=token_data["token_uri"],
        client_id=token_data["client_id"],
        client_secret=token_data["client_secret"],
        scopes=SCOPES,
    )


def _refresh_if_needed(creds):
    """Refresh credentials if expired, update session. Returns True if valid."""
    if creds.valid:
        return True
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        session["google_token"] = creds.to_json()
        return True
    return False


@app.route("/api/auth/status")
def auth_status():
    """Check if valid Google Calendar credentials exist."""
    # Check session cookie
    token_str = session.get("google_token")
    if token_str:
        try:
            token_data = json.loads(token_str)
            creds = _load_creds_from_token_data(token_data)
            if _refresh_if_needed(creds):
                return jsonify({"authenticated": True})
        except Exception:
            pass

    # Check token.json file (local dev)
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
            if creds.valid or (creds.expired and creds.refresh_token):
                if creds.expired:
                    creds.refresh(Request())
                # Migrate file token to session
                session["google_token"] = creds.to_json()
                return jsonify({"authenticated": True})
        except Exception:
            pass

    # Check GOOGLE_TOKEN env var (legacy Vercel support)
    token_env = os.environ.get("GOOGLE_TOKEN")
    if token_env:
        try:
            token_data = json.loads(token_env)
            creds = _load_creds_from_token_data(token_data)
            if _refresh_if_needed(creds):
                return jsonify({"authenticated": True})
        except Exception:
            pass

    return jsonify({"authenticated": False})


def _get_redirect_uri():
    """Get the OAuth redirect URI, preferring the one registered in credentials."""
    client_config = _get_oauth_client_config()
    if client_config:
        # Use the first registered redirect URI from the credentials file
        key = "web" if "web" in client_config else "installed"
        uris = client_config.get(key, {}).get("redirect_uris", [])
        if uris:
            return uris[0]
    return request.url_root.rstrip("/") + "/api/auth/callback"


@app.route("/api/auth/login")
def auth_login():
    """Redirect to Google OAuth consent screen."""
    from google_auth_oauthlib.flow import Flow

    client_config = _get_oauth_client_config()
    if not client_config:
        return jsonify({"error": "No OAuth credentials configured. Add credentials.json or set GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET env vars."}), 500

    redirect_uri = _get_redirect_uri()

    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=redirect_uri)
    auth_url, state = flow.authorization_url(access_type="offline", prompt="consent")
    # Store the code_verifier for PKCE — needed in callback
    session["oauth_state"] = state
    session["code_verifier"] = flow.code_verifier
    return redirect(auth_url)


@app.route("/api/auth/callback")
def auth_callback():
    """Handle OAuth callback, save token, redirect to app."""
    from google_auth_oauthlib.flow import Flow

    client_config = _get_oauth_client_config()
    if not client_config:
        return jsonify({"error": "No OAuth credentials configured."}), 500

    redirect_uri = _get_redirect_uri()
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=redirect_uri)
    # Restore PKCE code_verifier from session
    flow.code_verifier = session.pop("code_verifier", None)

    # Replace 127.0.0.1 with localhost in the callback URL to match the registered redirect URI
    auth_response = request.url
    if "127.0.0.1" in redirect_uri:
        auth_response = auth_response.replace("://localhost:", "://127.0.0.1:")
    else:
        auth_response = auth_response.replace("://127.0.0.1:", "://localhost:")
    flow.fetch_token(authorization_response=auth_response)

    creds = flow.credentials
    session["google_token"] = creds.to_json()

    # Also save to file for local dev convenience
    try:
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    except OSError:
        pass  # Serverless — can't write files, session cookie is enough

    return redirect("/")


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    """Clear token so user can re-authenticate."""
    session.pop("google_token", None)
    try:
        if os.path.exists(TOKEN_PATH):
            os.remove(TOKEN_PATH)
    except OSError:
        pass
    return jsonify({"status": "ok"})

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

tools = [
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


def _get_token_str():
    """Get token string from session, file, or env. Call within request context."""
    token_str = session.get("google_token")
    if token_str:
        return token_str

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH) as f:
            return f.read()

    token_env = os.environ.get("GOOGLE_TOKEN")
    if token_env:
        return token_env

    return None


def get_calendar_service(token_str=None):
    from googleapiclient.discovery import build

    creds = None

    if not token_str:
        token_str = _get_token_str()

    if token_str:
        token_data = json.loads(token_str)
        creds = _load_creds_from_token_data(token_data)
    else:
        raise Exception("No Google Calendar credentials found. Please reconnect.")

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return build("calendar", "v3", credentials=creds)


def create_calendar_event(summary, start_datetime, end_datetime, description="", location="", _token=None):
    service = get_calendar_service(token_str=_token)
    event = {
        "summary": summary,
        "start": {"dateTime": start_datetime, "timeZone": TIMEZONE},
        "end": {"dateTime": end_datetime, "timeZone": TIMEZONE},
    }
    if description:
        event["description"] = description
    if location:
        event["location"] = location

    created = service.events().insert(calendarId="primary", body=event).execute()
    return {
        "status": "created",
        "summary": created["summary"],
        "start": created["start"].get("dateTime"),
        "end": created["end"].get("dateTime"),
        "link": created.get("htmlLink", ""),
    }


def list_upcoming_events(days_ahead=7, _token=None):
    service = get_calendar_service(token_str=_token)
    now = datetime.datetime.utcnow()
    time_min = now.isoformat() + "Z"
    time_max = (now + datetime.timedelta(days=days_ahead)).isoformat() + "Z"

    result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        maxResults=20,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = result.get("items", [])
    return {
        "events": [
            {
                "event_id": e["id"],
                "summary": e.get("summary", "No title"),
                "start": e["start"].get("dateTime", e["start"].get("date")),
                "end": e["end"].get("dateTime", e["end"].get("date")),
                "location": e.get("location", ""),
            }
            for e in events
        ]
    }


def delete_calendar_event(event_id, summary="", _token=None):
    service = get_calendar_service(token_str=_token)
    try:
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return {
            "status": "deleted",
            "event_id": event_id,
            "summary": summary,
        }
    except Exception as e:
        return {
            "status": "error",
            "event_id": event_id,
            "summary": summary,
            "error": str(e),
        }


FUNC_MAP = {
    "create_calendar_event": create_calendar_event,
    "list_upcoming_events": list_upcoming_events,
    "delete_calendar_event": delete_calendar_event,
}


def sse(event_type, data):
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


def streaming_response(generator):
    """Create an SSE response with headers to prevent buffering on Vercel."""
    return Response(
        generator,
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def build_system_message():
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    return {"role": "system", "content": SYSTEM_PROMPT.format(today=today, tz=TIMEZONE)}


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
    token_str = _get_token_str()  # Extract before generator to avoid request context issue

    def generate():
      try:
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )

        content = ""
        collected_tc = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                content += delta.content
                yield sse("token", {"content": delta.content})

            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in collected_tc:
                        collected_tc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_chunk.id:
                        collected_tc[idx]["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name:
                            collected_tc[idx]["name"] += tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            collected_tc[idx]["arguments"] += tc_chunk.function.arguments

        if collected_tc:
            # Group tool calls by type
            creates = []
            deletes = []
            lists = []
            for idx in sorted(collected_tc):
                tc = collected_tc[idx]
                if tc["name"] == "create_calendar_event":
                    creates.append(tc)
                elif tc["name"] == "delete_calendar_event":
                    deletes.append(tc)
                elif tc["name"] == "list_upcoming_events":
                    lists.append(tc)

            # Build single history entry with ALL tool calls
            all_tcs = creates + deletes + lists
            all_tc_list = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in all_tcs
            ]
            tc_history_entry = {
                "role": "assistant",
                "content": content,
                "tool_calls": all_tc_list,
            }
            updated_history = history + [tc_history_entry]

            # Emit create confirmations
            if len(creates) == 1:
                tc = creates[0]
                args = json.loads(tc["arguments"])
                yield sse("confirm", {
                    "event": args,
                    "tool_call_id": tc["id"],
                    "tool_call": tc,
                    "history": updated_history,
                })
            elif len(creates) > 1:
                items = []
                for tc in creates:
                    args = json.loads(tc["arguments"])
                    items.append({"event": args, "tool_call_id": tc["id"], "tool_call": tc})
                yield sse("confirm_batch", {
                    "items": items,
                    "history": updated_history,
                })

            # Emit delete confirmations
            if len(deletes) == 1:
                tc = deletes[0]
                args = json.loads(tc["arguments"])
                yield sse("confirm_delete", {
                    "event": args,
                    "tool_call_id": tc["id"],
                    "tool_call": tc,
                    "history": updated_history,
                })
            elif len(deletes) > 1:
                items = []
                for tc in deletes:
                    args = json.loads(tc["arguments"])
                    items.append({"event": args, "tool_call_id": tc["id"], "tool_call": tc})
                yield sse("confirm_delete_batch", {
                    "items": items,
                    "history": updated_history,
                })

            # Auto-execute list_upcoming_events
            for tc in lists:
                args = json.loads(tc["arguments"])
                result = list_upcoming_events(**args, _token=token_str)

                tool_result_entry = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                }
                updated_history = updated_history + [tool_result_entry]
                follow_messages = [sys_msg] + updated_history[-20:]

                yield sse("tool_used", {"name": "list_upcoming_events"})

                follow_stream = client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=follow_messages,
                    tools=tools,
                    stream=True,
                )
                follow_content = ""
                follow_tc = {}
                for fchunk in follow_stream:
                    fdelta = fchunk.choices[0].delta
                    if fdelta.content:
                        follow_content += fdelta.content
                        yield sse("token", {"content": fdelta.content})
                    if fdelta.tool_calls:
                        for tc_chunk in fdelta.tool_calls:
                            fidx = tc_chunk.index
                            if fidx not in follow_tc:
                                follow_tc[fidx] = {"id": "", "name": "", "arguments": ""}
                            if tc_chunk.id:
                                follow_tc[fidx]["id"] = tc_chunk.id
                            if tc_chunk.function:
                                if tc_chunk.function.name:
                                    follow_tc[fidx]["name"] += tc_chunk.function.name
                                if tc_chunk.function.arguments:
                                    follow_tc[fidx]["arguments"] += tc_chunk.function.arguments

                if follow_tc:
                    # Process tool calls from the follow-up stream
                    follow_creates = []
                    follow_deletes = []
                    for fidx in sorted(follow_tc):
                        ftc = follow_tc[fidx]
                        if ftc["name"] == "create_calendar_event":
                            follow_creates.append(ftc)
                        elif ftc["name"] == "delete_calendar_event":
                            follow_deletes.append(ftc)

                    all_follow_tc_list = [
                        {
                            "id": ftc["id"],
                            "type": "function",
                            "function": {"name": ftc["name"], "arguments": ftc["arguments"]},
                        }
                        for ftc in list(follow_tc.values())
                    ]
                    follow_history_entry = {
                        "role": "assistant",
                        "content": follow_content,
                        "tool_calls": all_follow_tc_list,
                    }
                    updated_history.append(follow_history_entry)

                    if len(follow_creates) == 1:
                        ftc = follow_creates[0]
                        fargs = json.loads(ftc["arguments"])
                        yield sse("confirm", {
                            "event": fargs, "tool_call_id": ftc["id"],
                            "tool_call": ftc, "history": updated_history,
                        })
                    elif len(follow_creates) > 1:
                        items = [{"event": json.loads(ftc["arguments"]), "tool_call_id": ftc["id"], "tool_call": ftc} for ftc in follow_creates]
                        yield sse("confirm_batch", {"items": items, "history": updated_history})

                    if len(follow_deletes) == 1:
                        ftc = follow_deletes[0]
                        fargs = json.loads(ftc["arguments"])
                        yield sse("confirm_delete", {
                            "event": fargs, "tool_call_id": ftc["id"],
                            "tool_call": ftc, "history": updated_history,
                        })
                    elif len(follow_deletes) > 1:
                        items = [{"event": json.loads(ftc["arguments"]), "tool_call_id": ftc["id"], "tool_call": ftc} for ftc in follow_deletes]
                        yield sse("confirm_delete_batch", {"items": items, "history": updated_history})
                else:
                    updated_history.append({"role": "assistant", "content": follow_content})
                    yield sse("done", {"history": updated_history})
        else:
            history.append({"role": "assistant", "content": content})
            yield sse("done", {"history": history})
      except Exception as e:
        yield sse("token", {"content": f"Error: {str(e)}"})
        yield sse("done", {"history": history})

    return streaming_response(generate())


@app.route("/api/confirm", methods=["POST"])
def confirm_event():
    """User confirmed — actually execute the tool call and stream GPT's follow-up."""
    data = request.json
    history = data.get("history", [])
    tool_call = data.get("tool_call")
    args = json.loads(tool_call["arguments"])

    token_str = _get_token_str()
    func = FUNC_MAP[tool_call["name"]]
    result = func(**args, _token=token_str)

    history.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": json.dumps(result),
    })

    sys_msg = build_system_message()
    messages = [sys_msg] + history[-20:]

    def generate():
        yield sse("tool_used", {"name": tool_call["name"]})

        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            tools=tools,
            stream=True,
        )
        content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                yield sse("token", {"content": delta.content})

        history.append({"role": "assistant", "content": content})
        yield sse("done", {"history": history})

    return streaming_response(generate())


@app.route("/api/confirm_batch", methods=["POST"])
def confirm_batch():
    """Execute multiple accepted tool calls and stream GPT's follow-up."""
    data = request.json
    history = data.get("history", [])
    accepted = data.get("tool_calls", [])
    rejected = data.get("rejected_tool_calls", [])
    token_str = _get_token_str()

    for tc in accepted:
        args = json.loads(tc["arguments"])
        func = FUNC_MAP[tc["name"]]
        result = func(**args, _token=token_str)
        history.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps(result),
        })

    for tc in rejected:
        history.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps({"status": "cancelled", "reason": "User declined."}),
        })

    sys_msg = build_system_message()
    messages = [sys_msg] + history[-20:]

    def generate():
        yield sse("tool_used", {"name": accepted[0]["name"] if accepted else "batch"})

        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            tools=tools,
            stream=True,
        )
        content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                yield sse("token", {"content": delta.content})

        history.append({"role": "assistant", "content": content})
        yield sse("done", {"history": history})

    return streaming_response(generate())


@app.route("/api/reject_batch", methods=["POST"])
def reject_batch():
    """Reject all tool calls in a batch."""
    data = request.json
    history = data.get("history", [])
    tool_calls = data.get("tool_calls", [])

    for tc in tool_calls:
        history.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps({"status": "cancelled", "reason": "User declined."}),
        })

    sys_msg = build_system_message()
    messages = [sys_msg] + history[-20:]

    def generate():
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            tools=tools,
            stream=True,
        )
        content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                yield sse("token", {"content": delta.content})

        history.append({"role": "assistant", "content": content})
        yield sse("done", {"history": history})

    return streaming_response(generate())


@app.route("/api/reject", methods=["POST"])
def reject_event():
    """User rejected the event — tell GPT it was cancelled."""
    data = request.json
    history = data.get("history", [])
    tool_call = data.get("tool_call")

    history.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": json.dumps({"status": "cancelled", "reason": "User declined to create this event."}),
    })

    sys_msg = build_system_message()
    messages = [sys_msg] + history[-20:]

    def generate():
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            tools=tools,
            stream=True,
        )
        content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                yield sse("token", {"content": delta.content})

        history.append({"role": "assistant", "content": content})
        yield sse("done", {"history": history})

    return streaming_response(generate())
