"""Agent 4: Contextual Reasoning Agent.
Agent 5: Q&A Agent.

The reasoning agent synthesizes across document-understanding metadata and
policy-analysis output to decide if the question is fully answered or needs
another retrieval pass. The QA agent produces the final, user-facing answer.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.config import settings

REASONING_PROMPT = """You are a Contextual Reasoning Agent. You receive:
1. Document metadata (type, key fields, summary) from the Document Understanding Agent.
2. Policy analysis output (with tool-calling results and cited sources) from the \
Policy Analysis Agent.
3. The user's original question.

Decide: is the available information sufficient to answer the question fully and \
accurately? Respond ONLY with valid JSON:
{"sufficient": true/false, "reasoning": "...", "follow_up_query": "..." }
"follow_up_query" should be a refined search query ONLY if sufficient is false, \
otherwise null."""

QA_PROMPT = """You are the Q&A Agent, the final voice of the insurance document \
intelligence system. Using the document metadata, policy analysis, and reasoning \
provided, write a clear, direct answer for the end user (an insurance professional \
or policyholder). Cite source documents/clauses briefly. If information is missing, \
say so plainly instead of speculating. Keep it concise and well-structured."""


def run_contextual_reasoning(doc_meta: dict, analysis: dict, question: str) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model, api_key=settings.openai_api_key, temperature=0
    )
    import json

    content = (
        f"Document metadata:\n{json.dumps(doc_meta, indent=2)}\n\n"
        f"Policy analysis:\n{json.dumps(analysis, indent=2)}\n\n"
        f"User question:\n{question}"
    )
    messages = [SystemMessage(content=REASONING_PROMPT), HumanMessage(content=content)]
    response = llm.invoke(messages)
    raw = response.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"sufficient": True, "reasoning": raw, "follow_up_query": None}


def run_qa(doc_meta: dict, analysis: dict, reasoning: dict, question: str) -> str:
    llm = ChatOpenAI(
        model=settings.openai_model, api_key=settings.openai_api_key, temperature=0.2
    )
    import json

    content = (
        f"Document metadata:\n{json.dumps(doc_meta, indent=2)}\n\n"
        f"Policy analysis:\n{json.dumps(analysis, indent=2)}\n\n"
        f"Reasoning:\n{json.dumps(reasoning, indent=2)}\n\n"
        f"User question:\n{question}"
    )
    messages = [SystemMessage(content=QA_PROMPT), HumanMessage(content=content)]
    response = llm.invoke(messages)
    return response.content
