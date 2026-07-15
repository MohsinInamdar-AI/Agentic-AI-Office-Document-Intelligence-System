"""Loads insurance documents (PDF/DOCX/TXT) and splits them into semantic chunks."""
from __future__ import annotations

import os
import uuid
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def _load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in (".txt", ".md"):
        return _load_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_document(
    text: str,
    filename: str,
    doc_type: str = "policy_document",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Tuple[str, List[str], List[dict]]:
    """Returns (doc_id, chunks, metadatas) ready for vector store ingestion."""
    doc_id = uuid.uuid4().hex
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "doc_type": doc_type,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    return doc_id, chunks, metadatas
