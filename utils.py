# utils.py

"""Utility functions for document processing and vector store operations.

This module provides:
- `add_documents` – splits a document into chunks and adds them to the Chroma vector store.
- `retrieve_from_store` – performs a similarity search on the stored embeddings.

The implementation uses LangChain's `OpenAIEmbeddings` when an OpenAI API key is
available; otherwise it falls back to a simple deterministic embedding based on the
UTF‑8 bytes of the text. This ensures the code runs even without external API
access (useful for local testing).
"""

import os
from typing import List

from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# Attempt to import OpenAI embeddings; fallback to a dummy implementation
try:
    from langchain.embeddings import OpenAIEmbeddings
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None

from config import SETTINGS

# -----------------------------------------------------------------------
# Embedding provider
# -----------------------------------------------------------------------

def _get_embedding_model() -> Embeddings:
    """Return an Embeddings instance.

    If an OpenAI API key is set, use OpenAIEmbeddings; otherwise use a very
    lightweight deterministic embedding that hashes the text. This dummy model
    complies with the LangChain interface (embeds a list of texts into a list of
    vectors of equal length).
    """
    if SETTINGS.OPENAI_API_KEY:
        if OpenAIEmbeddings is None:
            raise RuntimeError("OpenAIEmbeddings import failed despite API key.")
        return OpenAIEmbeddings(openai_api_key=SETTINGS.OPENAI_API_KEY)

    # Simple fallback embedding – each character's ordinal value summed and
    # repeated to create a fixed‑size vector (dim=5). This is *not* meaningful but
    # satisfies the vectorstore API for offline usage.
    class DummyEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]):
            return [self._embed(t) for t in texts]

        def embed_query(self, text: str):
            return self._embed(text)

        @staticmethod
        def _embed(text: str):
            # Produce a deterministic 5‑dim vector
            base = sum(ord(c) for c in text) % 1000
            return [float(base + i) for i in range(5)]

    return DummyEmbeddings()

# -----------------------------------------------------------------------
# Vector store handling
# -----------------------------------------------------------------------

def _get_vector_store() -> Chroma:
    """Instantiate (or load) the Chroma vector store.

    The store persists under ``SETTINGS.VECTOR_STORE_PATH``. If the directory does
    not exist it will be created automatically.
    """
    embedding = _get_embedding_model()
    # Ensure the directory exists
    os.makedirs(SETTINGS.VECTOR_STORE_PATH, exist_ok=True)
    return Chroma(persist_directory=SETTINGS.VECTOR_STORE_PATH, embedding_function=embedding)

# -----------------------------------------------------------------------
# Document chunking
# -----------------------------------------------------------------------

def _split_text(text: str) -> List[str]:
    """Split a large text into manageable chunks.

    Uses LangChain's ``RecursiveCharacterTextSplitter`` with the same defaults as
    the original ``get_relevant_chunks`` implementation (chunk size 1500, overlap
    300). The splitter returns a list of chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    return splitter.split_text(text)

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def add_documents(text: str, source_name: str) -> None:
    """Add a document's chunks to the vector store.

    Parameters
    ----------
    text: str
        Full text content of the document.
    source_name: str
        Identifier used as ``metadata['source']`` for each chunk. Helps with
        provenance when displaying search results.
    """
    chunks = _split_text(text)
    store = _get_vector_store()
    # Prepare documents with metadata
    docs = [
        {
            "page_content": chunk,
            "metadata": {"source": source_name, "chunk_index": i},
        }
        for i, chunk in enumerate(chunks)
    ]
    store.add_documents(docs)
    store.persist()


def retrieve_from_store(query: str, k: int = 5) -> List[str]:
    """Retrieve the most relevant chunks for a query.

    Returns only the raw text of each chunk, ordered by relevance.
    """
    store = _get_vector_store()
    results = store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
