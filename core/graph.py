"""LangGraph orchestration wiring together all agents into a stateful workflow.

Graph shape:

    retrieve -> analyze -> reason --(insufficient)--> retrieve (refined query)
                              |
                          (sufficient)
                              v
                             qa -> END
"""
from __future__ import annotations

import contextvars
import time
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.retrieval_agent import retrieve_context
from agents.policy_analysis_agent import run_policy_analysis
from agents.reasoning_qa_agent import run_contextual_reasoning, run_qa
from core.evaluator import score_faithfulness
from core.metrics_store import metrics_store, RunRecord

# Holds the RunRecord for the pipeline invocation currently in flight on this
# call stack, so node functions can log timing/metrics without threading the
# record through LangGraph's state (which is meant for agent data, not
# observability plumbing).
_current_record: contextvars.ContextVar[Optional[RunRecord]] = contextvars.ContextVar(
    "current_record", default=None
)


class GraphState(TypedDict, total=False):
    question: str
    doc_id: Optional[str]
    doc_meta: dict
    context_chunks: list
    analysis: dict
    reasoning: dict
    answer: str
    loop_count: int


MAX_LOOPS = 2


def retrieve_node(state: GraphState) -> GraphState:
    record = _current_record.get()
    query = state.get("reasoning", {}).get("follow_up_query") or state["question"]
    with metrics_store.time_node(record, "retrieval_agent") if record else _noop():
        chunks = retrieve_context(query=query, doc_id=state.get("doc_id"))
    if record:
        record.retrieval_scores = [c.get("score", 0) for c in chunks]
        record.chunk_count = len(chunks)
    return {"context_chunks": chunks}


def analyze_node(state: GraphState) -> GraphState:
    record = _current_record.get()
    with metrics_store.time_node(record, "policy_analysis_agent") if record else _noop():
        analysis = run_policy_analysis(state["question"], state["context_chunks"])
    if record:
        record.tool_calls.extend(tc["tool"] for tc in analysis.get("tool_calls", []))
    return {"analysis": analysis}


def reason_node(state: GraphState) -> GraphState:
    record = _current_record.get()
    with metrics_store.time_node(record, "contextual_reasoning_agent") if record else _noop():
        reasoning = run_contextual_reasoning(
            state.get("doc_meta", {}), state["analysis"], state["question"]
        )
    loop_count = state.get("loop_count", 0)
    return {"reasoning": reasoning, "loop_count": loop_count + 1}


def qa_node(state: GraphState) -> GraphState:
    record = _current_record.get()
    with metrics_store.time_node(record, "qa_agent") if record else _noop():
        answer = run_qa(
            state.get("doc_meta", {}), state["analysis"], state["reasoning"], state["question"]
        )
    return {"answer": answer}


from contextlib import contextmanager


@contextmanager
def _noop():
    yield


def route_after_reasoning(state: GraphState) -> str:
    sufficient = state.get("reasoning", {}).get("sufficient", True)
    if sufficient or state.get("loop_count", 0) >= MAX_LOOPS:
        return "qa"
    return "retrieve"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("reason", reason_node)
    graph.add_node("qa", qa_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "reason")
    graph.add_conditional_edges(
        "reason", route_after_reasoning, {"qa": "qa", "retrieve": "retrieve"}
    )
    graph.add_edge("qa", END)

    return graph.compile()


compiled_graph = build_graph()


def run_qa_pipeline(question: str, doc_id: Optional[str] = None, doc_meta: Optional[dict] = None) -> dict:
    record = metrics_store.new_run(question=question, doc_id=doc_id)
    token = _current_record.set(record)
    start = time.perf_counter()
    try:
        initial_state: GraphState = {
            "question": question,
            "doc_id": doc_id,
            "doc_meta": doc_meta or {},
            "loop_count": 0,
        }
        final_state = compiled_graph.invoke(initial_state)
        record.loop_count = final_state.get("loop_count", 0)
        answer = final_state.get("answer") or ""
        record.answer_length = len(answer)

        faithfulness = score_faithfulness(final_state.get("context_chunks", []), answer)
        record.faithfulness_score = faithfulness.get("score")
        record.faithfulness_verdict = faithfulness.get("verdict")

        return {
            "answer": answer,
            "analysis": final_state.get("analysis"),
            "reasoning": final_state.get("reasoning"),
            "sources": final_state.get("analysis", {}).get("sources", []),
            "eval": {
                "run_id": record.run_id,
                "faithfulness_score": record.faithfulness_score,
                "faithfulness_verdict": record.faithfulness_verdict,
                "loop_count": record.loop_count,
                "avg_retrieval_score": (
                    round(sum(record.retrieval_scores) / len(record.retrieval_scores), 3)
                    if record.retrieval_scores else None
                ),
            },
        }
    except Exception as e:
        record.success = False
        record.error = str(e)
        raise
    finally:
        record.total_latency_ms = round((time.perf_counter() - start) * 1000, 2)
        metrics_store.commit(record)
        metrics_store.print_run(record)
        metrics_store.print_summary()
        _current_record.reset(token)
