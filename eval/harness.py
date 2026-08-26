"""
Offline evaluation harness against the synthetic golden incident dataset.

Runs the REAL pipeline (triage_node -> diagnose_node -> plan_action_node)
against every incident in data/synthetic/golden_incidents.jsonl and scores
it against known ground truth. This calls the real Anthropic API once per
incident by default — use --limit to run a quick subset while iterating.

Requires ANTHROPIC_API_KEY (same as Stage 4). Without it, diagnose_node
fails closed on every case (confidence 0.0, escalate) — the harness still
runs and produces a report, it'll just correctly show ~0% accuracy, which
is expected fail-closed behavior, not a bug in the harness.

Usage:
    python3 eval/harness.py                   # full 60-incident run
    python3 eval/harness.py --limit 10         # quick 10-incident sample
    python3 eval/harness.py --pattern schema_break   # just one pattern

Metrics tracked (first-class deliverable, not bolted on afterward):
  - RCA top-1 / top-3 accuracy: does true_cause match hypothesis rank 1 / any of 1-3?
    Matching is keyword-overlap (Jaccard) similarity, NOT exact string
    match — real model phrasing never matches ground truth verbatim.
  - Confidence calibration: bucket predicted confidence vs empirical
    accuracy in that bucket. A well-calibrated system's 0.8-0.9-confidence
    bucket should be right roughly 80-90% of the time.
  - False-action rate: does plan_action_node's classification of the
    predicted cause map to the same ActionType as the ground-truth action?
  - Faithfulness proxy: word-overlap between each hypothesis's reasoning
    text and the evidence it actually cited. This is a CHEAP STAND-IN for
    real RAGAS/DeepEval faithfulness scoring (see eval/faithfulness.py) —
    good enough to catch a hypothesis whose reasoning doesn't reference
    its own cited evidence at all, not a substitute for the real metric
    before this graduates past personal-project status.

CI regression gate: fail the build if any metric drops below its threshold.
See .github/workflows/eval.yml for how this plugs into CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.faithfulness import faithfulness_proxy  # noqa: E402
from graph.nodes import diagnose_node, plan_action_node, triage_node  # noqa: E402
from mcp_servers.runbook_kb.server import _tokenize  # noqa: E402  (reused for consistency)
from schemas.models import (  # noqa: E402
    ActionType,
    BreakSeverity,
    DiagnosisResult,
    ProposedAction,
    ReconBreak,
)

GOLDEN_PATH = ROOT / "data" / "synthetic" / "golden_incidents.jsonl"
REPORT_PATH = ROOT / "eval" / "last_report.json"

THRESHOLDS = {
    "rca_top1_accuracy": 0.60,
    "rca_top3_accuracy": 0.85,
    "false_action_rate_max": 0.25,
    "faithfulness_min": 0.50,
}

# Below this token-overlap ratio, a predicted cause is NOT considered a
# match for the ground-truth cause. Deliberately lenient — real model
# phrasing and the synthetic ground-truth string share very few exact
# words even when substantively correct (see the BRK-TM-012 live example
# from Stage 4: "settlement batch completed before TXN-42768 was posted"
# vs ground truth "settlement batch ran before gateway feed settled" —
# genuinely correct, ~30% token overlap).
CAUSE_MATCH_THRESHOLD = 0.15

CONFIDENCE_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]


@dataclass
class EvalCase:
    break_id: str
    pattern: str
    true_cause: str
    true_action: str
    break_event: ReconBreak
    predicted: DiagnosisResult | None = None
    predicted_action: ProposedAction | None = None
    evidence_by_id: dict | None = None
    error: str | None = None


@dataclass
class EvalReport:
    n_cases: int = 0
    rca_top1_correct: int = 0
    rca_top3_correct: int = 0
    false_actions: int = 0
    n_scored_for_action: int = 0
    calibration: dict = field(default_factory=dict)
    faithfulness_scores: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def rca_top1_accuracy(self) -> float:
        return self.rca_top1_correct / self.n_cases if self.n_cases else 0.0

    @property
    def rca_top3_accuracy(self) -> float:
        return self.rca_top3_correct / self.n_cases if self.n_cases else 0.0

    @property
    def false_action_rate(self) -> float:
        return self.false_actions / self.n_scored_for_action if self.n_scored_for_action else 0.0

    @property
    def avg_faithfulness(self) -> float:
        return sum(self.faithfulness_scores) / len(self.faithfulness_scores) \
            if self.faithfulness_scores else 0.0

    def passes_ci_gate(self) -> bool:
        return (
            self.rca_top1_accuracy >= THRESHOLDS["rca_top1_accuracy"]
            and self.rca_top3_accuracy >= THRESHOLDS["rca_top3_accuracy"]
            and self.false_action_rate <= THRESHOLDS["false_action_rate_max"]
            and self.avg_faithfulness >= THRESHOLDS["faithfulness_min"]
        )

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "rca_top1_accuracy": round(self.rca_top1_accuracy, 4),
            "rca_top3_accuracy": round(self.rca_top3_accuracy, 4),
            "false_action_rate": round(self.false_action_rate, 4),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "calibration": self.calibration,
            "passes_ci_gate": self.passes_ci_gate(),
            "thresholds": THRESHOLDS,
            "errors": self.errors,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"EVAL REPORT — {self.n_cases} incidents")
        print(f"{'=' * 60}")
        print(f"RCA top-1 accuracy:   {self.rca_top1_accuracy:.1%}  "
              f"(threshold: {THRESHOLDS['rca_top1_accuracy']:.0%})")
        print(f"RCA top-3 accuracy:   {self.rca_top3_accuracy:.1%}  "
              f"(threshold: {THRESHOLDS['rca_top3_accuracy']:.0%})")
        print(f"False-action rate:    {self.false_action_rate:.1%}  "
              f"(threshold: <={THRESHOLDS['false_action_rate_max']:.0%})")
        print(f"Avg faithfulness*:    {self.avg_faithfulness:.1%}  "
              f"(threshold: {THRESHOLDS['faithfulness_min']:.0%})")
        print("\nConfidence calibration:")
        for label, stats in self.calibration.items():
            if stats["n"] == 0:
                continue
            acc = stats["n_correct"] / stats["n"]
            print(f"  {label:>12}: n={stats['n']:>3}  "
                  f"avg_confidence={stats['avg_conf']:.2f}  "
                  f"actual_accuracy={acc:.1%}")
        if self.errors:
            print(f"\n{len(self.errors)} case(s) errored during evaluation:")
            for e in self.errors[:5]:
                print(f"  - {e}")
        print(f"\nCI gate: {'PASS' if self.passes_ci_gate() else 'FAIL'}")
        print("* faithfulness is a lightweight word-overlap proxy, not full "
              "RAGAS/DeepEval scoring — see eval/faithfulness.py")


def load_golden_dataset(path: str | Path = GOLDEN_PATH,
                         pattern_filter: str | None = None,
                         limit: int | None = None) -> list[EvalCase]:
    cases = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if pattern_filter and row["pattern"] != pattern_filter:
                continue
            be = row["break_event"]
            brk = ReconBreak(
                break_id=be["break_id"],
                detected_at=datetime.fromisoformat(be["detected_at"]),
                source_system=be["source_system"],
                counter_system=be["counter_system"],
                txn_ref=be["txn_ref"],
                amount_mismatch=be["amount_mismatch"],
                description=be["description"],
                severity=BreakSeverity(be["severity"]),
            )
            cases.append(EvalCase(
                break_id=row["break_event"]["break_id"],
                pattern=row["pattern"],
                true_cause=row["true_cause"],
                true_action=row["true_action"],
                break_event=brk,
            ))
    if limit:
        cases = cases[:limit]
    return cases


def _cause_matches(predicted_cause: str, true_cause: str) -> bool:
    """Keyword-overlap (Jaccard) similarity — real model phrasing rarely
    matches ground truth verbatim even when substantively correct. See
    module docstring for a live example of a correct-but-differently
    phrased match. Threshold tuned loosely; tighten it once you have
    enough real eval runs to know your false-positive rate.
    """
    pred_tokens = _tokenize(predicted_cause)
    true_tokens = _tokenize(true_cause)
    if not pred_tokens or not true_tokens:
        return False
    overlap = pred_tokens & true_tokens
    union = pred_tokens | true_tokens
    score = len(overlap) / len(union) if union else 0.0
    return score >= CAUSE_MATCH_THRESHOLD


def run_case(case: EvalCase) -> None:
    """Runs the real pipeline (TRIAGE -> DIAGNOSE -> PLAN ACTION) on one
    case, mutating it in place with predicted/predicted_action/error.
    """
    try:
        state = {"break_event": case.break_event}
        state = triage_node(state)
        case.evidence_by_id = {item.evidence_id: item.content for item in state["evidence"].items}
        state = diagnose_node(state)
        case.predicted = state["diagnosis"]
        state = plan_action_node(state)
        case.predicted_action = state["proposed_action"]
    except Exception as e:
        case.error = f"{type(e).__name__}: {e}"


def _confidence_bucket_label(confidence: float) -> str:
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= confidence < hi:
            return f"[{lo:.1f}-{min(hi, 1.0):.1f}]"
    return "unknown"


def score(cases: list[EvalCase]) -> EvalReport:
    report = EvalReport(n_cases=len(cases))
    report.calibration = {
        _confidence_bucket_label(lo): {"n": 0, "n_correct": 0, "conf_sum": 0.0, "avg_conf": 0.0}
        for lo, hi in CONFIDENCE_BUCKETS
    }

    for case in cases:
        if case.error:
            report.errors.append(f"{case.break_id}: {case.error}")
            continue
        if case.predicted is None or not case.predicted.hypotheses:
            report.errors.append(f"{case.break_id}: no prediction produced")
            continue

        hyps = case.predicted.hypotheses
        top1_correct = _cause_matches(hyps[0].cause, case.true_cause)
        top3_correct = any(_cause_matches(h.cause, case.true_cause) for h in hyps)
        if top1_correct:
            report.rca_top1_correct += 1
        if top3_correct:
            report.rca_top3_correct += 1

        bucket = _confidence_bucket_label(hyps[0].confidence)
        if bucket in report.calibration:
            report.calibration[bucket]["n"] += 1
            report.calibration[bucket]["conf_sum"] += hyps[0].confidence
            if top1_correct:
                report.calibration[bucket]["n_correct"] += 1

        if case.predicted_action is not None:
            report.n_scored_for_action += 1
            try:
                true_action_enum = ActionType(case.true_action)
            except ValueError:
                true_action_enum = None
            if true_action_enum and case.predicted_action.action_type != true_action_enum:
                report.false_actions += 1

        report.faithfulness_scores.append(
            faithfulness_proxy(case.predicted, evidence_by_id=case.evidence_by_id)
        )

    for bucket, stats in report.calibration.items():
        stats["avg_conf"] = stats["conf_sum"] / stats["n"] if stats["n"] else 0.0

    return report


def main():
    parser = argparse.ArgumentParser(description="Recon Sentinel eval harness")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N incidents (fast iteration)")
    parser.add_argument("--pattern", type=str, default=None,
                         help="Only run incidents of this pattern (e.g. schema_break)")
    args = parser.parse_args()

    cases = load_golden_dataset(pattern_filter=args.pattern, limit=args.limit)
    print(f"Running {len(cases)} incident(s) through the real pipeline "
          f"(TRIAGE -> DIAGNOSE -> PLAN ACTION)...")
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case.break_id} ({case.pattern})...", end=" ", flush=True)
        run_case(case)
        if case.error:
            print(f"ERROR: {case.error}")
        else:
            top = case.predicted.hypotheses[0] if case.predicted.hypotheses else None
            print(f"confidence={top.confidence:.2f}" if top else "no hypothesis")

    report = score(cases)
    report.print_summary()

    with open(REPORT_PATH, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nFull report written to {REPORT_PATH}")

    if not report.passes_ci_gate():
        sys.exit(1)


if __name__ == "__main__":
    main()
