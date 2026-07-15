"""FastAPI service exposing the Agentic AI Insurance Document Intelligence platform."""
from __future__ import annotations

import os
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import settings
from core.document_loader import load_document, chunk_document
from core.vectorstore import vector_store
from agents.document_understanding_agent import run_document_understanding
from core.graph import run_qa_pipeline
from core.metrics_store import metrics_store

app = FastAPI(
    title="Agentic AI Office Document Intelligence System (Insurance)",
    description=(
        "Multi-agent platform for insurance document understanding, contextual "
        "reasoning, policy analysis, and Q&A, built with LangGraph, LangChain, "
        "OpenAI, and ChromaDB."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory registry of ingested documents' metadata (swap for a DB in production).
DOCUMENT_REGISTRY: dict[str, dict] = {}


class QARequest(BaseModel):
    question: str
    doc_id: Optional[str] = None


class QAResponse(BaseModel):
    answer: str
    sources: list
    tool_calls: list = []
    reasoning: dict = {}
    eval: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "vector_count": vector_store.count()}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Agent 1 entry point: ingest, classify, chunk, and embed a document."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported file type: {ext}")

    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = load_document(temp_path)
        if not text.strip():
            raise HTTPException(422, "No extractable text found in document.")

        doc_meta = run_document_understanding(text, file.filename)
        doc_id, chunks, metadatas = chunk_document(
            text, filename=file.filename, doc_type=doc_meta.get("doc_type", "other")
        )
        vector_store.add_chunks(chunks, metadatas, doc_id=doc_id)

        DOCUMENT_REGISTRY[doc_id] = {
            "filename": file.filename,
            **doc_meta,
            "chunk_count": len(chunks),
        }

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            **doc_meta,
        }
    finally:
        os.remove(temp_path)


@app.get("/documents")
def list_documents():
    return DOCUMENT_REGISTRY


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    if doc_id not in DOCUMENT_REGISTRY:
        raise HTTPException(404, "Document not found")
    return DOCUMENT_REGISTRY[doc_id]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in DOCUMENT_REGISTRY:
        raise HTTPException(404, "Document not found")
    vector_store.delete_document(doc_id)
    del DOCUMENT_REGISTRY[doc_id]
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/qa", response_model=QAResponse)
def question_answer(req: QARequest):
    """Runs the full multi-agent LangGraph pipeline: retrieve -> analyze -> reason -> answer."""
    doc_meta = DOCUMENT_REGISTRY.get(req.doc_id, {}) if req.doc_id else {}
    result = run_qa_pipeline(question=req.question, doc_id=req.doc_id, doc_meta=doc_meta)
    return QAResponse(
        answer=result["answer"] or "",
        sources=result["sources"],
        tool_calls=result.get("analysis", {}).get("tool_calls", []),
        reasoning=result.get("reasoning", {}),
        eval=result.get("eval", {}),
    )


@app.get("/metrics/summary")
def metrics_summary():
    """Aggregate agentic-AI evaluation metrics across all runs this session:
    per-agent latency, retrieval quality, tool-call rate, self-correction loop
    count, and LLM-as-judge faithfulness scores."""
    return metrics_store.summary()


@app.get("/metrics/history")
def metrics_history(limit: int = 50):
    """Per-run breakdown, most recent first."""
    return metrics_store.history(limit=limit)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.app_host, port=settings.app_port, reload=True)
