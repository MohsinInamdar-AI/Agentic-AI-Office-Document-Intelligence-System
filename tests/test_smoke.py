"""Basic smoke tests that don't require an OpenAI key (structure/import checks)."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_imports():
    from core import config, document_loader
    from agents import tools

    assert config.settings is not None
    assert callable(document_loader.chunk_document)
    assert len(tools.ALL_TOOLS) == 3


def test_chunking():
    from core.document_loader import chunk_document

    text = "Insurance clause. " * 500
    doc_id, chunks, metas = chunk_document(text, filename="test.txt")
    assert doc_id
    assert len(chunks) > 1
    assert len(chunks) == len(metas)


def test_tool_logic():
    from agents.tools import compute_premium_breakdown, check_coverage_sufficiency

    result = compute_premium_breakdown.invoke({"annual_premium": 1200, "payment_frequency": "monthly"})
    assert result["installment_amount"] == 100.0

    result2 = check_coverage_sufficiency.invoke({"claim_amount": 5000, "coverage_limit": 3000})
    assert result2["fully_covered"] is False
    assert result2["shortfall"] == 2000


if __name__ == "__main__":
    test_imports()
    test_chunking()
    test_tool_logic()
    print("All smoke tests passed.")
