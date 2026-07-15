"""Agent 3: Policy Analysis Agent.

Reasons over retrieved policy clauses and, where numeric/rule-based logic is
needed (premium math, coverage checks, expiration dates), invokes structured
tools rather than letting the LLM guess.
"""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from core.config import settings
from agents.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a Policy Analysis Agent for insurance documents. You are \
given retrieved policy clauses as context and a specific analysis request. Analyze \
coverage terms, exclusions, limits, and obligations precisely, citing which clause \
supports each claim you make. When a question requires arithmetic (premium math, \
coverage-vs-claim comparisons, date/expiration math), you MUST call the appropriate \
tool instead of computing it yourself. Be conservative: if the context does not \
contain enough information, say so explicitly rather than guessing."""


def run_policy_analysis(question: str, context_chunks: list[dict]) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    ).bind_tools(ALL_TOOLS)

    context_text = "\n\n".join(
        f"[Source: {c['metadata'].get('filename', 'unknown')} | chunk "
        f"{c['metadata'].get('chunk_index')}]\n{c['text']}"
        for c in context_chunks
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Retrieved context:\n{context_text}\n\nAnalysis request:\n{question}"
        ),
    ]

    response = llm.invoke(messages)
    messages.append(response)

    tool_calls_made = []
    if response.tool_calls:
        tool_map = {t.name: t for t in ALL_TOOLS}
        for call in response.tool_calls:
            tool_fn = tool_map.get(call["name"])
            if not tool_fn:
                continue
            result = tool_fn.invoke(call["args"])
            tool_calls_made.append({"tool": call["name"], "args": call["args"], "result": result})
            messages.append(
                ToolMessage(content=json.dumps(result), tool_call_id=call["id"])
            )
        # Second pass: let the model incorporate tool results into a final answer.
        response = llm.invoke(messages)

    return {
        "analysis": response.content,
        "tool_calls": tool_calls_made,
        "sources": [
            {
                "filename": c["metadata"].get("filename"),
                "chunk_index": c["metadata"].get("chunk_index"),
                "score": round(c.get("score", 0), 3),
            }
            for c in context_chunks
        ],
    }
