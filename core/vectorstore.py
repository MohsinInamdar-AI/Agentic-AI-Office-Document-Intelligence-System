"""ChromaDB vector store wrapper for semantic retrieval over insurance documents."""
from __future__ import annotations

import uuid
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings

from core.config import settings


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection with OpenAI embeddings."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.chroma_db_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    def add_chunks(
        self,
        texts: List[str],
        metadatas: List[dict],
        doc_id: Optional[str] = None,
    ) -> List[str]:
        if not texts:
            return []
        vectors = self._embeddings.embed_documents(texts)
        ids = [f"{doc_id or uuid.uuid4().hex}-{i}" for i in range(len(texts))]
        self._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = None,
        where: Optional[dict] = None,
    ) -> List[dict]:
        k = k or settings.max_retrieval_k
        query_vec = self._embeddings.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where,
        )
        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"text": doc, "metadata": meta, "score": 1 - dist})
        return hits

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._collection.count()


vector_store = VectorStore()
