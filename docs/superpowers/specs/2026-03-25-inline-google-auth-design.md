# Inline Google OAuth Login

## Problem

CalBot requires running a separate `setup_auth.py` script to authenticate with Google Calendar. When the token expires, the user must re-run the script manually. The auth flow should be part of the web UI.

## Design

### Auth Flow

1. On page load, frontend calls `GET /api/auth/status`
2. If no valid token: show login overlay with "Connect Google Calendar" button
3. Button navigates to `GET /api/auth/login` → Google OAuth consent → callback saves `token.json`
4. On success: redirect to `/`, auth check passes, chat UI loads
5. Header gains a "Reconnect" button to clear token and re-auth when needed

### Backend Endpoints

**`GET /api/auth/status`** — Returns `{"authenticated": true/false}`. Checks if credentials exist (file or env var) and are valid. Attempts refresh if expired.

**`GET /api/auth/login`** — Reads OAuth client config from `credentials.json` or `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` env vars. Builds Google OAuth URL with `redirect_uri` pointing to `/api/auth/callback`. Redirects user to Google consent screen.

**`GET /api/auth/callback`** — Receives auth code from Google, exchanges for tokens, saves to `token.json`. Redirects to `/`.

**`POST /api/auth/logout`** — Deletes `token.json`. Returns `{"status": "ok"}`.

### Frontend Changes

- Add login overlay: centered CalBot branding + calendar icon + "Connect Google Calendar" button
- On load: fetch `/api/auth/status`, show login or chat accordingly
- Add "Reconnect" button in header (visible only when authenticated)
- Login overlay styled consistently with existing dark/amber theme

### Credential Sources (priority order)

1. `token.json` file (local dev, written by callback)
2. `GOOGLE_TOKEN` env var (Vercel deployment)

### OAuth Client Config Sources

1. `credentials.json` file (local dev)
2. `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` env vars (Vercel)

### Scope

- No changes to chat, confirm, or reject endpoints
- No multi-user support
- `setup_auth.py` remains for users who prefer CLI setup
