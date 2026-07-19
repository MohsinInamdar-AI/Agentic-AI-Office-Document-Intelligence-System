
# Agentic AI Office Document Intelligence System (Insurance)

Enterprise-grade multi-agent platform for insurance document analysis. Built with
**LangGraph** (agent orchestration), **LangChain** + **OpenAI** (LLM reasoning &
embeddings), **ChromaDB** (semantic retrieval), and **FastAPI** (service layer).

## Architecture

```
Upload ──▶ [Document Understanding Agent] ──▶ chunk + embed ──▶ ChromaDB
                                                                    │
User Q ──▶ [Retrieval Agent] ◀────────────────────────────────────┘
              │  (semantic search, k-NN over embeddings)
              ▼
        [Policy Analysis Agent] ──▶ Tool Calling (premium math,
              │                      coverage checks, expiration dates)
              ▼
      [Contextual Reasoning Agent] ──(insufficient context)──▶ back to Retrieval
              │ (sufficient)
              ▼
          [Q&A Agent] ──▶ Final answer + cited sources
```

This flow is implemented as a LangGraph `StateGraph` in `core/graph.py` with a
bounded self-correction loop (max 2 retrieval refinements) so the reasoning agent
can ask the retrieval agent for a better query before answering.

## Agents

| Agent                  | File                                                            | Responsibility                                              |
| ---------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| Document Understanding | `agents/document_understanding_agent.py`                      | Classifies doc type, extracts policy fields, summarizes     |
| Retrieval              | `agents/retrieval_agent.py`                                   | Semantic search over ChromaDB (RAG)                         |
| Policy Analysis        | `agents/policy_analysis_agent.py`                             | Clause-level reasoning + OpenAI tool calling for math/rules |
| Contextual Reasoning   | `agents/reasoning_qa_agent.py` (`run_contextual_reasoning`) | Judges answer sufficiency, triggers re-retrieval            |
| Q&A                    | `agents/reasoning_qa_agent.py` (`run_qa`)                   | Produces final cited answer                                 |

## Tool Calling

`agents/tools.py` defines deterministic tools bound to the Policy Analysis Agent:

- `compute_premium_breakdown` — installment math
- `check_coverage_sufficiency` — claim vs. coverage-limit checks
- `days_until_expiration` — policy expiration/date math

These prevent the LLM from hallucinating numbers — any arithmetic is delegated to
real code.

## Frontend

A simple standalone frontend is included at `frontend/index.html` — no build step,
no dependencies. It lets you upload documents, see extracted metadata, and chat
with the Q&A pipeline (with cited sources and tool calls shown inline).

1. Start the backend (`uvicorn api.main:app --reload`).
2. Open `frontend/index.html` directly in your browser (double-click it, or
   `open frontend/index.html` / drag into a browser tab).
3. Confirm the "API base URL" field at the top matches where your backend is
   running (default `http://localhost:8000`).

CORS is enabled on the backend (`api/main.py`) so the static HTML file can call
the API directly without needing to be served from the same origin.

## Setup

```bash
cp .env.example .env        # add your OPENAI_API_KEY
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Or with Docker:

```bash
cp .env.example .env        # add your OPENAI_API_KEY
docker compose up --build
```

API docs available at `http://localhost:8000/docs`.

## API

- `POST /documents/upload` — upload a PDF/TXT policy/claim document; runs the
  Document Understanding Agent, chunks, and embeds into ChromaDB.
- `GET /documents` — list ingested documents and their extracted metadata.
- `DELETE /documents/{doc_id}` — remove a document and its vectors.
- `POST /qa` — `{"question": "...", "doc_id": "optional-scope-to-one-doc"}` runs
  the full LangGraph pipeline and returns an answer with cited sources and any
  tool calls used.
- `GET /health` — liveness + vector store count.

## Example

```bash
curl -X POST http://localhost:8000/documents/upload -F "file=@sample_policy.pdf"

curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the coverage limit for water damage, and is a $12,000 claim fully covered?"}'
```

## Notes / Next Steps

- `DOCUMENT_REGISTRY` is in-memory — swap for Postgres/Redis in production.
- Add auth (API keys/OAuth) before exposing beyond a trusted network.
- For very large corpora, consider hybrid search (BM25 + embeddings) in
  `core/vectorstore.py`.
- Add streaming responses (`langgraph` supports `.stream()`) for a chat-style UI.

# Agentic-AI-Office-Document-Intelligence-System

Enterprise Agentic AI platform for insurance document analysis using multi-agent workflows (LangGraph, LangChain, FastAPI, ChromaDB)
