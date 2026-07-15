"""LLM-as-judge evaluator: scores whether the final answer is actually grounded
in the retrieved context (faithfulness / groundedness) rather than hallucinated.

This is one of the standard RAG evaluation metrics (see RAGAS-style faithfulness)
and is computed as a cheap, separate LLM call after the main pipeline finishes.
"""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.config import settings

JUDGE_PROMPT = """You are an evaluation judge for a RAG system. Given the retrieved \
context and the generated answer, score how faithful/grounded the answer is in the \
context on a 0.0-1.0 scale:
- 1.0 = every claim in the answer is directly supported by the context
- 0.5 = partially supported, some claims not traceable to context
- 0.0 = answer is unsupported or contradicts the context

Respond ONLY with valid JSON: {"score": 0.0-1.0, "verdict": "one short sentence"}"""


def score_faithfulness(context_chunks: list[dict], answer: str) -> dict:
    if not answer or not context_chunks:
        return {"score": None, "verdict": "insufficient data to judge"}

    llm = ChatOpenAI(
        model=settings.openai_model, api_key=settings.openai_api_key, temperature=0
    )
    context_text = "\n\n".join(c.get("text", "") for c in context_chunks)[:8000]
    messages = [
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=f"Context:\n{context_text}\n\nAnswer:\n{answer}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        return {"score": float(parsed.get("score")), "verdict": parsed.get("verdict")}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"score": None, "verdict": raw[:200]}
