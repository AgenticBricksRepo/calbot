"""Local dev server — run with: uv run python app.py"""

import os
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

from dotenv import load_dotenv
load_dotenv()

from api.index import app

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)
