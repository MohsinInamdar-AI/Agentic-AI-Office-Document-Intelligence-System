"""Evaluation & observability layer for the multi-agent pipeline.

Tracks the metrics that matter for agentic AI systems in an interview /
demo context:

- Per-agent latency (Document Understanding, Retrieval, Policy Analysis,
  Contextual Reasoning, Q&A)
- Retrieval quality: top-k similarity scores, avg score, chunk count
- Self-correction behavior: how many retrieval loops the Reasoning Agent
  triggered before it was satisfied
- Tool-calling: how often and which deterministic tools were invoked
  (signal that the system offloads math instead of hallucinating it)
- Groundedness / faithfulness: LLM-as-judge score of whether the final
  answer is actually supported by the retrieved context
- End-to-end latency and running success/failure counts

Everything is stored in-memory (swap for a time-series DB in production)
and exposed via /metrics/summary and /metrics/history in api/main.py, and
visualized at /dashboard.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional


@dataclass
class RunRecord:
    run_id: str
    question: str
    doc_id: Optional[str]
    started_at: float
    node_latencies_ms: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0
    retrieval_scores: list = field(default_factory=list)
    chunk_count: int = 0
    loop_count: int = 0
    tool_calls: list = field(default_factory=list)
    faithfulness_score: Optional[float] = None
    faithfulness_verdict: Optional[str] = None
    answer_length: int = 0
    success: bool = True
    error: Optional[str] = None


class MetricsStore:
    """Rolling in-memory store of the last N pipeline runs."""

    def __init__(self, max_history: int = 200) -> None:
        self._runs: deque[RunRecord] = deque(maxlen=max_history)

    def new_run(self, question: str, doc_id: Optional[str]) -> RunRecord:
        record = RunRecord(
            run_id=uuid.uuid4().hex[:8],
            question=question,
            doc_id=doc_id,
            started_at=time.time(),
        )
        return record

    def commit(self, record: RunRecord) -> None:
        self._runs.append(record)

    @contextmanager
    def time_node(self, record: RunRecord, node_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            record.node_latencies_ms[node_name] = round(
                record.node_latencies_ms.get(node_name, 0) + elapsed_ms, 2
            )

    def all_runs(self) -> list[RunRecord]:
        return list(self._runs)

    def summary(self) -> dict:
        runs = self.all_runs()
        if not runs:
            return {
                "total_runs": 0,
                "success_rate": None,
                "avg_total_latency_ms": None,
                "avg_node_latency_ms": {},
                "avg_retrieval_score": None,
                "avg_chunk_count": None,
                "avg_loop_count": None,
                "tool_call_rate": None,
                "avg_faithfulness_score": None,
                "tool_usage_breakdown": {},
            }

        node_names = set()
        for r in runs:
            node_names.update(r.node_latencies_ms.keys())
        avg_node_latency = {
            n: round(mean(r.node_latencies_ms.get(n, 0) for r in runs), 2)
            for n in node_names
        }

        all_scores = [s for r in runs for s in r.retrieval_scores]
        faithfulness_scores = [r.faithfulness_score for r in runs if r.faithfulness_score is not None]
        tool_usage: dict[str, int] = {}
        for r in runs:
            for tc in r.tool_calls:
                tool_usage[tc] = tool_usage.get(tc, 0) + 1

        runs_with_tools = sum(1 for r in runs if r.tool_calls)

        return {
            "total_runs": len(runs),
            "success_rate": round(sum(r.success for r in runs) / len(runs), 3),
            "avg_total_latency_ms": round(mean(r.total_latency_ms for r in runs), 2),
            "avg_node_latency_ms": avg_node_latency,
            "avg_retrieval_score": round(mean(all_scores), 3) if all_scores else None,
            "avg_chunk_count": round(mean(r.chunk_count for r in runs), 2),
            "avg_loop_count": round(mean(r.loop_count for r in runs), 2),
            "tool_call_rate": round(runs_with_tools / len(runs), 3),
            "avg_faithfulness_score": round(mean(faithfulness_scores), 3) if faithfulness_scores else None,
            "tool_usage_breakdown": tool_usage,
        }

    def history(self, limit: int = 50) -> list[dict]:
        runs = self.all_runs()[-limit:]
        return [
            {
                "run_id": r.run_id,
                "question": r.question,
                "doc_id": r.doc_id,
                "total_latency_ms": round(r.total_latency_ms, 2),
                "node_latencies_ms": r.node_latencies_ms,
                "avg_retrieval_score": round(mean(r.retrieval_scores), 3) if r.retrieval_scores else None,
                "chunk_count": r.chunk_count,
                "loop_count": r.loop_count,
                "tool_calls": r.tool_calls,
                "faithfulness_score": r.faithfulness_score,
                "faithfulness_verdict": r.faithfulness_verdict,
                "success": r.success,
                "error": r.error,
            }
            for r in reversed(runs)
        ]


    def print_run(self, record: RunRecord) -> None:
        """Pretty-prints a single run's metrics to the terminal (uvicorn console)."""
        bar = "=" * 70
        lines = [
            "",
            bar,
            f"[EVAL] Run {record.run_id}  |  success={record.success}",
            bar,
            f"Question         : {record.question}",
            f"Doc scope        : {record.doc_id or 'all documents'}",
            "-" * 70,
            "Per-agent latency (ms):",
        ]
        for node, ms in record.node_latencies_ms.items():
            lines.append(f"    {node:<28} {ms:>10.2f} ms")
        lines.append(f"    {'TOTAL':<28} {record.total_latency_ms:>10.2f} ms")
        lines.append("-" * 70)
        avg_score = (
            round(sum(record.retrieval_scores) / len(record.retrieval_scores), 3)
            if record.retrieval_scores else None
        )
        lines.append(f"Retrieval        : {record.chunk_count} chunks | avg similarity score = {avg_score}")
        lines.append(f"Self-correction  : {record.loop_count} reasoning loop(s)")
        lines.append(f"Tool calls       : {record.tool_calls or 'none'}")
        lines.append(f"Faithfulness     : score={record.faithfulness_score}  verdict=\"{record.faithfulness_verdict}\"")
        lines.append(f"Answer length    : {record.answer_length} chars")
        if record.error:
            lines.append(f"Error            : {record.error}")
        lines.append(bar)
        lines.append("")
        print("\n".join(lines))

    def print_summary(self) -> None:
        """Pretty-prints the aggregate summary across all runs to the terminal."""
        s = self.summary()
        bar = "=" * 70
        print("\n" + bar)
        print("[EVAL] AGGREGATE METRICS SUMMARY")
        print(bar)
        if s["total_runs"] == 0:
            print("No runs recorded yet.")
            print(bar + "\n")
            return
        print(f"Total runs               : {s['total_runs']}")
        print(f"Success rate             : {s['success_rate']}")
        print(f"Avg total latency (ms)   : {s['avg_total_latency_ms']}")
        print("Avg per-agent latency (ms):")
        for node, ms in s["avg_node_latency_ms"].items():
            print(f"    {node:<28} {ms:>10.2f} ms")
        print(f"Avg retrieval score      : {s['avg_retrieval_score']}")
        print(f"Avg chunks retrieved     : {s['avg_chunk_count']}")
        print(f"Avg self-correction loops: {s['avg_loop_count']}")
        print(f"Tool call rate           : {s['tool_call_rate']}")
        print(f"Avg faithfulness score   : {s['avg_faithfulness_score']}")
        print(f"Tool usage breakdown     : {s['tool_usage_breakdown']}")
        print(bar + "\n")


metrics_store = MetricsStore()
