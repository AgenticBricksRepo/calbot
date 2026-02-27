"""
Run this ONCE locally to get your Google Calendar token.
It will print out the GOOGLE_TOKEN value to paste into Vercel env vars.

Prerequisites:
  1. Go to https://console.cloud.google.com/
  2. Create a project and enable Google Calendar API
  3. Create OAuth 2.0 credentials (Desktop app type)
  4. Download the JSON and save as credentials.json in this directory
"""

import os
import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def main():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                print("ERROR: credentials.json not found!")
                return

            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as f:
            f.write(creds.to_json())

    # Build the env var value
    token_data = json.loads(creds.to_json())
    env_value = json.dumps(token_data)

    print()
    print("=" * 60)
    print("  SUCCESS! Copy the value below into Vercel.")
    print("=" * 60)
    print()
    print("Go to your Vercel project -> Settings -> Environment Variables")
    print("Add these two variables:")
    print()
    print("1) OPENAI_API_KEY = <your OpenAI API key>")
    print()
    print("2) GOOGLE_TOKEN =")
    print(env_value)
    print()
    print("3) TIMEZONE = America/New_York  (change to your timezone)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
