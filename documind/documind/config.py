"""Central settings for DocuMind. Every other file imports from here."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file and put its values into os.environ.
load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"          # put your PDFs here
CHROMA_DIR = BASE_DIR / "chroma_db"   # vector database is saved here

# --- Model -------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic:claude-sonnet-5-5")

# --- Chunking (used in ingest.py) --------------------------------------
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks

# --- Retrieval (used in tools.py) --------------------------------------
TOP_K = 4              # how many chunks to fetch per search
MAX_DISTANCE = 0.6     # chunks farther than this are treated as "not relevant"

# --- Agent safety (used in agent.py) -----------------------------------
MAX_STEPS = 6          # hard stop so the agent can never loop forever


def check_api_key() -> None:
    """Fail early if the key for the chosen provider is missing."""
    provider = MODEL_NAME.split(":")[0]
    key_names = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    key_name = key_names.get(provider)
    if key_name and not os.getenv(key_name):
        raise RuntimeError(f"{key_name} is missing. Add it to your .env file.")