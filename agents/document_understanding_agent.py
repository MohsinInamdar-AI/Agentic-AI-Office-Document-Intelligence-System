"""Agent 1: Document Understanding.

Classifies the document type (policy, claim, endorsement, exclusion rider, etc.)
and extracts structured header-level metadata before retrieval-based agents run.
"""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.config import settings

SYSTEM_PROMPT = """You are a Document Understanding Agent for an insurance document \
intelligence system. Given raw extracted text from an uploaded document, you must:
1. Classify the document type: one of [policy_document, claim_form, endorsement, \
exclusion_rider, underwriting_report, other].
2. Extract any of the following fields if present: policy_number, insured_name, \
policy_type, effective_date, expiration_date, premium_amount, coverage_limits.
3. Give a one-paragraph plain-language summary of what the document is.

Respond ONLY with valid JSON in this shape:
{
  "doc_type": "...",
  "fields": { "policy_number": "...", "insured_name": "...", ... },
  "summary": "..."
}
If a field is not found, omit it or use null. Do not include any text outside the JSON."""


def run_document_understanding(text: str, filename: str) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    # Truncate very long docs for the classification pass; full text is still
    # chunked and embedded separately for retrieval.
    excerpt = text[:12000]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Filename: {filename}\n\nDocument text:\n{excerpt}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"doc_type": "other", "fields": {}, "summary": raw[:500]}
    return parsed
