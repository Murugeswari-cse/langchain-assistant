# config.py

"""Project configuration settings.

This module centralizes configurable parameters such as model selections,
vector‑store paths, and feature toggles. Adjust the values as needed for
your environment.
"""

import os
from pathlib import Path

class Settings:
    # LLM configuration – defaults to the Groq model used in main.py
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Embedding model – OpenAI embeddings if key is present, else a simple fallback
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Vector store configuration
    VECTOR_STORE_PATH: str = str(Path(__file__).parent / "vector_store")
    USE_VECTOR_STORE: bool = True  # toggle to enable/disable vector store integration

    # Miscellaneous
    TEMPERATURE: float = 0.0

# Export a singleton for easy import
SETTINGS = Settings()
