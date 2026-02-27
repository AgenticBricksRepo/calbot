"""Local dev server — run with: uv run python app.py"""

from dotenv import load_dotenv
load_dotenv()

from api.index import app

if __name__ == "__main__":
    app.run(debug=True, port=5000)
