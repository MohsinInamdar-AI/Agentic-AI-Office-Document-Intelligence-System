"""Structured tools exposed to agents via OpenAI tool calling.

These represent deterministic, auditable operations that the LLM can invoke
instead of hallucinating numeric or rule-based answers.
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def compute_premium_breakdown(annual_premium: float, payment_frequency: str) -> dict:
    """Break an annual premium into installment amounts.

    Args:
        annual_premium: The total annual premium amount.
        payment_frequency: One of 'monthly', 'quarterly', 'semi_annual', 'annual'.
    """
    divisors = {"monthly": 12, "quarterly": 4, "semi_annual": 2, "annual": 1}
    n = divisors.get(payment_frequency.lower(), 1)
    installment = round(annual_premium / n, 2)
    return {
        "annual_premium": annual_premium,
        "payment_frequency": payment_frequency,
        "installments": n,
        "installment_amount": installment,
    }


@tool
def check_coverage_sufficiency(claim_amount: float, coverage_limit: float) -> dict:
    """Check whether a claim amount is within the policy's coverage limit.

    Args:
        claim_amount: The amount being claimed.
        coverage_limit: The maximum coverage limit stated in the policy.
    """
    shortfall = max(0.0, claim_amount - coverage_limit)
    return {
        "claim_amount": claim_amount,
        "coverage_limit": coverage_limit,
        "fully_covered": shortfall == 0,
        "shortfall": shortfall,
    }


@tool
def days_until_expiration(expiration_date: str, reference_date: str) -> dict:
    """Calculate days between a reference date and a policy expiration date.

    Args:
        expiration_date: ISO date string (YYYY-MM-DD) of policy expiration.
        reference_date: ISO date string (YYYY-MM-DD) to compare against, e.g. today.
    """
    from datetime import date

    exp = date.fromisoformat(expiration_date)
    ref = date.fromisoformat(reference_date)
    delta = (exp - ref).days
    return {
        "expiration_date": expiration_date,
        "reference_date": reference_date,
        "days_remaining": delta,
        "is_expired": delta < 0,
    }


ALL_TOOLS = [compute_premium_breakdown, check_coverage_sufficiency, days_until_expiration]
