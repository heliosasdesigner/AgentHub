from __future__ import annotations

from typing import List

from langchain.schema import Document


def load_faiss_retriever() -> object:
    """
    Placeholder loader for a Faiss-backed retriever.

    Replace with real Faiss index loading and return a LangChain retriever.
    """
    class _StubRetriever:
        def get_relevant_documents(self, _: str) -> List[Document]:
            return []

    return _StubRetriever()
