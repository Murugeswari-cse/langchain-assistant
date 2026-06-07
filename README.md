# LangChain Assistant

A Streamlit‑based AI assistant built with **LangChain**, **Groq**, and **Tavily** for web search. The app can ingest PDFs or text files, split them into chunks, and answer questions using a large language model.

## Features
- Document upload (PDF / TXT / MD)
- Keyword‑based chunk retrieval (`get_relevant_chunks`)
- Groq LLM with fallback models
- Clean configuration via `config.py`

## Setup

```bash
# 1️⃣ Clone the repo
git clone https://github.com/Murugeswari-cse/langchain-assistant.git
cd langchain-assistant

# 2️⃣ Create a virtual environment
python -m venv venv
.\\venv\\Scripts\\Activate.ps1   # PowerShell
# or: source venv/bin/activate  # Bash

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app
streamlit run main.py
```

## Configuration

Edit `config.py` to set your API keys and other flags:

```python
from config import SETTINGS

SETTINGS.GROQ_API_KEY = "your‑groq‑key"
SETTINGS.OPENAI_API_KEY = "your‑openai‑key"
SETTINGS.USE_VECTOR_STORE = False   # disabled per request
```

## License

MIT
