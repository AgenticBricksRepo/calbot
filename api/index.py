from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from openai import OpenAI
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import json
import os
import datetime

app = Flask(__name__, static_folder="../public")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── Config ────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
TIMEZONE = os.environ.get("TIMEZONE", "America/Los_Angeles")
SCOPES = ["https://www.googleapis.com/auth/calendar"]

SYSTEM_PROMPT = """You are CalBot, a sharp and friendly calendar assistant. You help the user manage their Google Calendar.

Today is {today}. The user's timezone is {tz}.

Rules:
- When the user wants to create an event, extract ALL details and call create_calendar_event.
- If they don't give a time, ask for it. Default duration is 1 hour.
- ALWAYS include a useful description for events. Summarize the purpose of the event in 1-2 sentences.
- When they ask about their schedule, call list_upcoming_events.
- Be concise. After creating events, confirm with the details.
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
            "description": "List upcoming events from Google Calendar.",
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
]


def get_calendar_service():
    from googleapiclient.discovery import build

    token_json = os.path.join(os.path.dirname(__file__), "..", "token.json")

    if os.path.exists(token_json):
        creds = Credentials.from_authorized_user_file(token_json, SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_json, "w") as f:
                f.write(creds.to_json())
    else:
        token_data = json.loads(os.environ["GOOGLE_TOKEN"])
        creds = Credentials(
            token=token_data["token"],
            refresh_token=token_data["refresh_token"],
            token_uri=token_data["token_uri"],
            client_id=token_data["client_id"],
            client_secret=token_data["client_secret"],
            scopes=SCOPES,
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())

    return build("calendar", "v3", credentials=creds)


def create_calendar_event(summary, start_datetime, end_datetime, description="", location=""):
    service = get_calendar_service()
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


def list_upcoming_events(days_ahead=7):
    service = get_calendar_service()
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
                "summary": e.get("summary", "No title"),
                "start": e["start"].get("dateTime", e["start"].get("date")),
                "end": e["end"].get("dateTime", e["end"].get("date")),
                "location": e.get("location", ""),
            }
            for e in events
        ]
    }


FUNC_MAP = {
    "create_calendar_event": create_calendar_event,
    "list_upcoming_events": list_upcoming_events,
}


def sse(event_type, data):
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


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

    def generate():
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
            for idx in sorted(collected_tc):
                tc = collected_tc[idx]

                if tc["name"] == "create_calendar_event":
                    # Send confirmation to frontend — don't execute yet
                    args = json.loads(tc["arguments"])
                    # Add assistant message with tool calls to history
                    tc_history_entry = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]},
                            }
                        ],
                    }
                    updated_history = history + [tc_history_entry]
                    yield sse("confirm", {
                        "event": args,
                        "tool_call_id": tc["id"],
                        "tool_call": tc,
                        "history": updated_history,
                    })

                elif tc["name"] == "list_upcoming_events":
                    args = json.loads(tc["arguments"])
                    result = list_upcoming_events(**args)

                    # Build history with tool call + result
                    tc_history_entry = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]},
                            }
                        ],
                    }
                    tool_result_entry = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result),
                    }
                    follow_history = history + [tc_history_entry, tool_result_entry]
                    follow_messages = [sys_msg] + follow_history[-20:]

                    # Stream the follow-up response
                    yield sse("tool_used", {"name": "list_upcoming_events"})

                    follow_stream = client.chat.completions.create(
                        model="gpt-5-nano",
                        messages=follow_messages,
                        tools=tools,
                        stream=True,
                    )
                    follow_content = ""
                    for fchunk in follow_stream:
                        fdelta = fchunk.choices[0].delta
                        if fdelta.content:
                            follow_content += fdelta.content
                            yield sse("token", {"content": fdelta.content})

                    follow_history.append({"role": "assistant", "content": follow_content})
                    yield sse("done", {"history": follow_history})
        else:
            history.append({"role": "assistant", "content": content})
            yield sse("done", {"history": history})

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/confirm", methods=["POST"])
def confirm_event():
    """User confirmed — actually create the event and stream GPT's follow-up."""
    data = request.json
    history = data.get("history", [])
    tool_call = data.get("tool_call")
    args = json.loads(tool_call["arguments"])

    result = create_calendar_event(**args)

    history.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": json.dumps(result),
    })

    sys_msg = build_system_message()
    messages = [sys_msg] + history[-20:]

    def generate():
        yield sse("tool_used", {"name": "create_calendar_event"})

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

    return Response(generate(), mimetype="text/event-stream")


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

    return Response(generate(), mimetype="text/event-stream")
