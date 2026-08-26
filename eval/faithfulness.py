"""
Faithfulness scoring for DiagnosisResult objects.

This module provides a LIGHTWEIGHT PROXY metric, not real RAGAS/DeepEval
faithfulness scoring. The distinction matters and shouldn't be glossed
over in an interview: this catches one specific, cheap-to-detect failure
mode — a hypothesis whose reasoning text doesn't actually reference the
evidence it claims to cite — via word overlap. It does NOT catch subtler
unfaithfulness (e.g. reasoning that mentions the right evidence but draws
an unsupported conclusion from it). That requires an LLM-as-judge
approach, which is what RAGAS's faithfulness metric and DeepEval's
FaithfulnessMetric actually do internally.

Real integration point (left as the natural next step once this system
needs to be more than a personal project's eval harness):

    from ragas import evaluate
    from ragas.metrics import faithfulness
    from datasets import Dataset

    rows = [{
        "question": break_description,
        "answer": hypothesis.reasoning,
        "contexts": [evidence_item.content for evidence_item in cited_items],
    } for ...]
    result = evaluate(Dataset.from_list(rows), metrics=[faithfulness])

That call costs an extra LLM call per hypothesis (RAGAS's faithfulness
metric is itself LLM-judged), which is why this proxy exists: it gives a
free, instant signal for CI-gate purposes on every commit, while the real
RAGAS run is better suited to a periodic/pre-release deeper eval pass
rather than every single PR.
"""
from __future__ import annotations

import re

from schemas.models import DiagnosisResult

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "of",
    "in", "on", "for", "with", "if", "be", "this", "that", "it", "as", "by",
    "at", "from", "has", "have", "had", "not", "no",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _hypothesis_faithfulness(reasoning: str, cited_evidence_content: list[str]) -> float:
    """Fraction of the reasoning's meaningful tokens that also appear
    somewhere in the cited evidence's text. A reasoning that's pure
    generic filler ("this is likely due to a system issue") with no
    actual overlap to its cited evidence scores near 0 here — exactly
    the "confident, fluent, ungrounded" failure mode this whole system
    exists to catch, applied reflexively to its own output.
    """
    reasoning_tokens = _tokenize(reasoning)
    if not reasoning_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for content in cited_evidence_content:
        evidence_tokens |= _tokenize(content)
    if not evidence_tokens:
        return 0.0
    overlap = reasoning_tokens & evidence_tokens
    return len(overlap) / len(reasoning_tokens)


def faithfulness_proxy(diagnosis: DiagnosisResult, evidence_by_id: dict[str, str] | None = None) -> float:
    """Average faithfulness proxy score across all hypotheses in a
    diagnosis. `evidence_by_id` maps evidence_id -> content text; if not
    provided, this can only check citation validity (already guaranteed
    by graph/llm_client.py) rather than reasoning-to-evidence overlap,
    and returns 1.0 for any hypothesis with at least one citation (a
    weaker but still meaningful signal: "did it cite anything at all").
    """
    if not diagnosis.hypotheses:
        return 0.0

    scores = []
    for hyp in diagnosis.hypotheses:
        if evidence_by_id:
            cited_content = [
                evidence_by_id[eid] for eid in hyp.cited_evidence_ids
                if eid in evidence_by_id
            ]
            scores.append(_hypothesis_faithfulness(hyp.reasoning, cited_content))
        else:
            scores.append(1.0 if hyp.cited_evidence_ids else 0.0)

    return sum(scores) / len(scores)
