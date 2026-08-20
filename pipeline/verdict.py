"""
Verdict Module
--------------
Generates a final verdict (REAL, LIKELY FAKE, or UNCERTAIN)
based on the aggregated score and stance counts.
"""


def generate_verdict(score: float, support_count: int, refute_count: int) -> dict:
    """
    Generate a verdict on the claim's truthfulness.

    Rules (based on [-1.0 to +1.0] score range):
        - REAL:        score >= 0.25 AND support_count >= refute_count AND support_count > 0
        - LIKELY FAKE: score <= -0.25 OR (refute_count > support_count * 1.5 AND refute_count > 0)
        - UNCERTAIN:   everything else (neutral/balanced evidence or score between -0.25 and +0.25)

    Args:
        score:         Aggregated credibility-weighted score (-1.0 to 1.0).
        support_count: Number of supporting evidence items.
        refute_count:  Number of refuting evidence items.

    Returns:
        Dict with verdict, confidence, and reasoning.
    """
    if support_count == 0 and refute_count == 0:
        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "reasoning": "No supporting or refuting evidence was found for the claim.",
        }

    if score >= 0.25 and support_count >= refute_count:
        verdict = "REAL"
        confidence = round(min(max((score + 1.0) / 2.0, 0.5), 1.0), 4)
        reasoning = (
            f"The claim is supported by {support_count} source(s) with a "
            f"positive aggregated score of {score:.2f}. Evidence aligns "
            f"with the claim."
        )

    elif score <= -0.25 or (refute_count > support_count * 1.5 and refute_count > 0):
        verdict = "LIKELY FAKE"
        confidence = round(min(max(abs(score - 1.0) / 2.0, 0.5), 1.0), 4)
        reasoning = (
            f"The claim is contradicted by {refute_count} source(s) with a "
            f"negative aggregated score of {score:.2f}. Evidence suggests the claim "
            f"is unreliable or false."
        )

    else:
        verdict = "UNCERTAIN"
        confidence = round(max(1.0 - abs(score), 0.3), 4)
        reasoning = (
            f"The evidence is mixed or neutral — {support_count} supporting and "
            f"{refute_count} refuting source(s) with a score of {score:.2f}. "
            f"Not enough evidence to make a definitive judgment."
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
    }
