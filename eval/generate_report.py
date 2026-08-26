"""
Generates a one-page, interview-ready markdown summary from an eval
report JSON (produced by eval/harness.py).

Usage:
    python3 eval/generate_report.py                    # uses eval/last_report.json
    python3 eval/generate_report.py --input path.json  # a specific report

Writes eval/EVAL_SUMMARY.md — print or screenshot this for interview prep,
or paste its contents into your portfolio writeup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "eval" / "last_report.json"
DEFAULT_OUTPUT = ROOT / "eval" / "EVAL_SUMMARY.md"


def _verdict(value: float, threshold: float, higher_is_better: bool = True) -> str:
    passed = (value >= threshold) if higher_is_better else (value <= threshold)
    return "✅ PASS" if passed else "⚠️  BELOW THRESHOLD"


def generate(report: dict) -> str:
    n = report["n_cases"]
    thresholds = report["thresholds"]
    lines = []
    lines.append("# Recon Sentinel — Evaluation Summary")
    lines.append("")
    lines.append(f"*Generated {report['generated_at']} · {n} incident(s) evaluated "
                  f"against the synthetic golden dataset*")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Result | Threshold | Verdict |")
    lines.append("|---|---|---|---|")
    lines.append(f"| RCA top-1 accuracy | {report['rca_top1_accuracy']:.1%} | "
                  f"≥{thresholds['rca_top1_accuracy']:.0%} | "
                  f"{_verdict(report['rca_top1_accuracy'], thresholds['rca_top1_accuracy'])} |")
    lines.append(f"| RCA top-3 accuracy | {report['rca_top3_accuracy']:.1%} | "
                  f"≥{thresholds['rca_top3_accuracy']:.0%} | "
                  f"{_verdict(report['rca_top3_accuracy'], thresholds['rca_top3_accuracy'])} |")
    lines.append(f"| False-action rate | {report['false_action_rate']:.1%} | "
                  f"≤{thresholds['false_action_rate_max']:.0%} | "
                  f"{_verdict(report['false_action_rate'], thresholds['false_action_rate_max'], higher_is_better=False)} |")
    lines.append(f"| Faithfulness (proxy) | {report['avg_faithfulness']:.1%} | "
                  f"≥{thresholds['faithfulness_min']:.0%} | "
                  f"{_verdict(report['avg_faithfulness'], thresholds['faithfulness_min'])} |")
    lines.append("")
    lines.append(f"**Overall CI gate: {'✅ PASS' if report['passes_ci_gate'] else '⚠️  FAIL'}**")
    lines.append("")

    lines.append("## Confidence calibration")
    lines.append("")
    lines.append("| Confidence bucket | n | Avg. confidence | Actual accuracy |")
    lines.append("|---|---|---|---|")
    for bucket, stats in report["calibration"].items():
        if stats["n"] == 0:
            continue
        acc = stats["n_correct"] / stats["n"]
        lines.append(f"| {bucket} | {stats['n']} | {stats['avg_conf']:.2f} | {acc:.1%} |")
    lines.append("")
    lines.append("A well-calibrated system's accuracy in each bucket should roughly "
                  "match the bucket's confidence range — e.g. the [0.7-0.9] bucket "
                  "landing around 70-90% actual accuracy.")
    lines.append("")

    if n < 30:
        lines.append("## A note on sample size")
        lines.append("")
        lines.append(f"This report reflects **{n} incidents**, not the full 60-incident "
                      f"golden dataset. At this sample size, a single incident's "
                      f"difference can swing top-3 accuracy by 10 percentage points — "
                      f"treat exact figures as directional, not final, until run "
                      f"against the full set (`python3 eval/harness.py` with no "
                      f"`--limit`).")
        lines.append("")

    if report["avg_faithfulness"] < thresholds["faithfulness_min"]:
        lines.append("## On the faithfulness score specifically")
        lines.append("")
        lines.append("The faithfulness metric here is a **lightweight word-overlap "
                      "proxy** (see `eval/faithfulness.py`), not full RAGAS/DeepEval "
                      "LLM-judged scoring. It catches pure fabrication well but "
                      "penalizes legitimate paraphrasing — a hypothesis that correctly "
                      "explains a break in its own words, without repeating the "
                      "evidence's exact phrasing, can score low here despite being "
                      "genuinely grounded. A below-threshold score is a signal to "
                      "spot-check reasoning manually, not proof of hallucination.")
        lines.append("")

    if report["errors"]:
        lines.append("## Errors during evaluation")
        lines.append("")
        for e in report["errors"][:10]:
            lines.append(f"- {e}")
        lines.append("")

    lines.append("## Interview talking points this supports")
    lines.append("")
    lines.append("- Evaluation was built as a first-class deliverable (RCA accuracy, "
                  "calibration, false-action rate, faithfulness) from Stage 7 onward, "
                  "not bolted on after the fact.")
    lines.append("- The CI gate is a real gate — it fires on real metric shortfalls "
                  "(see sample-size and faithfulness-proxy notes above for what a "
                  "genuine, explainable gate failure looks like versus a bug).")
    lines.append("- False-action rate is the metric that matters most for a "
                  "human-approval-gated system: even an imperfect diagnosis rarely "
                  "translates into a wrong *proposed action*, because plan_action_node's "
                  "classification is a separate, conservative, deterministic step.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"No report found at {args.input}. Run eval/harness.py first.")
        raise SystemExit(1)

    with open(args.input) as f:
        report = json.load(f)

    summary = generate(report)
    with open(args.output, "w") as f:
        f.write(summary)

    print(f"Wrote {args.output}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
