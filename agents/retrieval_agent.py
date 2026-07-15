"""Agent 2: Retrieval Agent.

Performs semantic retrieval over ChromaDB to fetch the most relevant chunks
for a given user query, optionally scoped to a specific document.
"""
from __future__ import annotations

from typing import List, Optional

from core.vectorstore import vector_store


def retrieve_context(
    query: str,
    doc_id: Optional[str] = None,
    k: Optional[int] = None,
) -> List[dict]:
    where = {"doc_id": doc_id} if doc_id else None
    return vector_store.similarity_search(query=query, k=k, where=where)
