import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


# Also expose module-level variables for direct imports if needed
GROQ_API_KEY = Config.GROQ_API_KEY
GROQ_MODEL = Config.GROQ_MODEL
CHROMA_PERSIST_DIR = Config.CHROMA_PERSIST_DIR
