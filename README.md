# CalBot

A chatbot that manages your Google Calendar through natural conversation. Powered by GPT-5 nano and the Google Calendar API.

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Set your OpenAI API key

Add your key to `.env`:

```
OPENAI_API_KEY=sk-...
TIMEZONE=America/Los_Angeles
```

### 3. Set up Google Calendar

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable the **Google Calendar API**
3. Go to **APIs & Services > OAuth consent screen**, set up consent, and publish the app
4. Go to **APIs & Services > Credentials**, create an **OAuth 2.0 Client ID** (Desktop app)
5. Download the JSON and save it as `credentials.json` in this directory

Then run the one-time auth:

```bash
uv run python setup_auth.py
```

Sign in with your Google account. This creates `token.json` which the app uses going forward.

### 4. Run locally

```bash
uv run python app.py
```

Open `http://localhost:5000`.

## Deploy to Vercel

Set these environment variables in your Vercel project:

| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `GOOGLE_TOKEN` | Contents of `token.json` (run `cat token.json`) |
| `TIMEZONE` | e.g. `America/Los_Angeles` |

Then deploy:

```bash
vercel
```

## Features

- Natural language event creation and schedule checking
- Streaming responses
- Confirmation prompt before creating events
- Mobile-friendly UI
