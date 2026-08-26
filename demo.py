"""
Recon Sentinel — live interview demo.

Fires one real seeded break through the actual compiled graph: real
evidence gathering, a real LLM diagnosis call, a genuine halt-and-resume
at the human-approval interrupt (you type y/n), and a real post-action
verification. Nothing in this script is mocked.

Usage:
    python3 demo.py                          # default: a timing-mismatch
                                               # break with a clean diagnosis
    python3 demo.py --break-id BRK-SB-001     # a specific seeded break
    python3 demo.py --reject                  # demo the rejection path
    python3 demo.py --auto-approve            # skip the y/n prompt (for
                                               # recording a screen capture
                                               # without live typing)

Requires ANTHROPIC_API_KEY set (see .env). If it's not set, DIAGNOSE will
correctly fail closed and escalate — which is itself a legitimate (if less
flashy) thing to show: "here's what it does when it doesn't have what it
needs, instead of guessing."
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from graph.build import graph  # noqa: E402
from graph.nodes import mint_approval_token  # noqa: E402
from schemas.models import ApprovalDecision, BreakSeverity, ReconBreak  # noqa: E402

GOLDEN_PATH = ROOT / "data" / "synthetic" / "golden_incidents.jsonl"

# A hand-picked default: known from a real Stage 4 run to produce a clean,
# well-cited, high-confidence diagnosis — good for a first demo run. Swap
# to --break-id BRK-SB-001 (schema_break) or BRK-DUP-001 (duplicate) to
# show the FLAG_FOR_REVIEW path instead of RERUN_JOB.
DEFAULT_BREAK_ID = "BRK-TM-012"


def _rule(char="─", width=70):
    print(char * width)


def _section(title: str):
    print()
    _rule("═")
    print(f"  {title}")
    _rule("═")


def _load_break(break_id: str) -> tuple[ReconBreak, dict]:
    with open(GOLDEN_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["break_event"]["break_id"] == break_id:
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
                return brk, row
    raise ValueError(f"break_id {break_id!r} not found in {GOLDEN_PATH}")


def _reset_break_state(brk: ReconBreak):
    """Undo a previous demo run's writes for this specific break, so you
    can rehearse the same break_id repeatedly. Safe because each seeded
    break in data/synthetic/generate.py gets its own unique txn_ref —
    deleting settlement rows for this txn_ref can't touch any other
    break's data.
    """
    import sqlite3
    db_path = ROOT / "data" / "synthetic" / "recon.db"
    with sqlite3.connect(db_path) as conn:
        # Only relevant if this break's fix was a rerun_job (which inserts
        # a settlement row that wasn't there in the original seed data).
        # For other action types (flag_for_review, mark_resolved, no_action)
        # this is a no-op, which is correct — nothing to undo.
        conn.execute("DELETE FROM settlement WHERE txn_ref = ?", (brk.txn_ref,))
        conn.execute("DELETE FROM episodic_memory WHERE break_id = ?", (brk.break_id,))
        conn.execute("DELETE FROM actions_taken WHERE break_id = ?", (brk.break_id,))
        conn.execute("DELETE FROM used_approval_tokens WHERE break_id = ?", (brk.break_id,))
        conn.execute("DELETE FROM review_queue WHERE break_id = ?", (brk.break_id,))
        conn.commit()
    print(f"  (--reset: cleared prior demo state for {brk.break_id} / {brk.txn_ref})")
    print(f"  NOTE: for timing_mismatch breaks this also removes the original seeded")
    print(f"  settlement row if one existed for another reason — re-run "
          f"data/synthetic/generate.py for a fully pristine reset if needed.")


def main():
    parser = argparse.ArgumentParser(description="Recon Sentinel live demo")
    parser.add_argument("--break-id", default=DEFAULT_BREAK_ID)
    parser.add_argument("--reject", action="store_true",
                         help="Demo the rejection path instead of approving")
    parser.add_argument("--auto-approve", action="store_true",
                         help="Skip the interactive y/n prompt")
    parser.add_argument("--reset", action="store_true",
                         help="Clear this break's prior demo writes first, so "
                              "you can rehearse the same break_id repeatedly")
    args = parser.parse_args()

    brk, golden_row = _load_break(args.break_id)

    if args.reset:
        _reset_break_state(brk)

    thread_id = f"demo-{args.break_id}-{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    _section("RECON SENTINEL — reconciliation break triage")
    print(f"  Break ID:     {brk.break_id}")
    print(f"  Pattern:      {golden_row['pattern']}  (ground truth, hidden from the system)")
    print(f"  Systems:      {brk.source_system}  <-->  {brk.counter_system}")
    print(f"  Transaction:  {brk.txn_ref}")
    print(f"  Description:  {brk.description}")
    print(f"  Severity:     {brk.severity.value}")

    _section("STAGE 1-3: MONITOR -> TRIAGE (deterministic evidence gathering)")
    for event in graph.stream({"break_event": brk}, config):
        for node, out in event.items():
            if node == "triage":
                evidence = out["evidence"]
                print(f"  Gathered {len(evidence.items)} evidence item(s) from the three "
                      f"read-only MCP servers:")
                for item in evidence.items[:6]:
                    print(f"    [{item.source_server}/{item.kind}] {item.content[:90]}")
                if len(evidence.items) > 6:
                    print(f"    ... and {len(evidence.items) - 6} more")
                if evidence.is_duplicate:
                    print("  -> Flagged as a DUPLICATE open incident.")
                if evidence.is_benign_known_pattern:
                    print("  -> Flagged as a BENIGN KNOWN PATTERN (matches a runbook).")

            elif node == "suppress":
                _section("SUPPRESSED")
                print("  This break was suppressed — duplicate or benign known pattern.")
                print("  No LLM call was made. This is the correct outcome for this case.")
                return

            elif node == "diagnose":
                diag = out["diagnosis"]
                _section("STAGE 4: DIAGNOSE (the one LLM call)")
                print(f"  Overall confidence: {diag.overall_confidence:.2f}")
                print(f"  Requires escalation: {diag.requires_escalation}")
                print()
                for h in diag.hypotheses:
                    print(f"  Hypothesis #{h.rank} (confidence={h.confidence:.2f}):")
                    print(f"    Cause:     {h.cause}")
                    print(f"    Cites:     {h.cited_evidence_ids}")
                    print(f"    Reasoning: {h.reasoning}")
                    print()

            elif node == "escalate":
                _section("ESCALATED")
                reason = out.get("escalation_reason", "unspecified")
                print(f"  Reason: {reason}")
                print("  This break was routed to a human review queue instead of")
                print("  proceeding on a low-confidence or ungrounded diagnosis.")
                if "ANTHROPIC_API_KEY" in str(reason):
                    print()
                    print("  (Set ANTHROPIC_API_KEY in your .env to see the full happy path.)")
                return

            elif node == "plan_action":
                action = out["proposed_action"]
                _section("STAGE 5: PLAN ACTION (deterministic classification)")
                print(f"  Proposed action:  {action.action_type.value}")
                print(f"  Target:           {action.target_ref}")
                print(f"  Justification:    {action.justification}")

    snapshot = graph.get_state(config)
    if not snapshot.next:
        print("\n(Graph finished without reaching approval — see ESCALATED above.)")
        return

    proposed = snapshot.values["proposed_action"]
    _section("STAGE 5 (cont.): HUMAN APPROVAL — genuine graph interrupt")
    print("  Execution is HALTED here. State is checkpointed. The graph will")
    print("  not proceed until an external approval event resumes this thread.")
    print()
    print(f"  Proposed action: {proposed.action_type.value} on {proposed.target_ref}")
    print(f"  Justification:   {proposed.justification}")
    print()

    if args.reject:
        decision = "rejected"
        print("  --reject flag set: simulating a human REJECTING this action.")
    elif args.auto_approve:
        decision = "approved"
        print("  --auto-approve flag set: simulating a human APPROVING this action.")
    else:
        answer = input("  Approve this action? [y/N]: ").strip().lower()
        decision = "approved" if answer == "y" else "rejected"

    if decision == "approved":
        token = mint_approval_token(brk.break_id, proposed.action_type, "demo.reviewer")
        approval = ApprovalDecision(break_id=brk.break_id, decision="approved", token=token)
    else:
        approval = ApprovalDecision(break_id=brk.break_id, decision="rejected",
                                     reviewer_note="Rejected in interview demo")

    _section("RESUMING — STAGE 5 (cont.): ACT / STAGE 6: VERIFY + REFLECT")
    graph.update_state(config, {"approval": approval})
    for event in graph.stream(None, config):
        for node, out in event.items():
            if node == "act":
                print(f"  [act] status={out.get('status')}")
            elif node == "escalate":
                print(f"  [escalate] reason={out.get('escalation_reason')}")
            elif node == "verify":
                v = out["verification"]
                print(f"  [verify] resolved={v.resolved}")
                print(f"           notes: {v.notes}")
            elif node == "reflect":
                print(f"  [reflect] outcome persisted to episodic memory.")
                print(f"            future similar breaks will retrieve this as context.")

    final = graph.get_state(config)
    _section("FINAL STATUS")
    print(f"  {final.values.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
