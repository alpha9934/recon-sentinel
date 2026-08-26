"""
Evidence gathering for TRIAGE.

Calls the three read-only MCP servers (ledger-telemetry, obs-context,
runbook-kb) and assembles their results into a single EvidenceBundle.

Stage 3 note: these are IN-PROCESS function calls (direct imports), not
calls over an actual MCP transport (stdio/SSE). That's a deliberate
simplification — the read/write trust-boundary split is still real
(this module only ever imports the three read servers, never
recon_actuator), it's just not yet running as separate OS processes.
Swapping to a real mcp.client session per server is a mechanical change
later; nothing about triage_node's logic needs to change when you do it.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from mcp_servers.ledger_telemetry import server as ledger_telemetry
from mcp_servers.obs_context import server as obs_context
from mcp_servers.runbook_kb import server as runbook_kb
from schemas.models import EvidenceBundle, EvidenceItem, ReconBreak

# How far around a break's detected_at to look for batch/log evidence.
EVIDENCE_WINDOW = timedelta(minutes=30)

# Deploys/schema-changes are frequently the root cause discovered well
# before the break surfaces (the synthetic schema_break pattern seeds
# its deploy event 1 hour before detection) — use a wider lookback here
# than for tight batch-timing evidence.
DEPLOY_LOOKBACK_WINDOW = timedelta(hours=4)

# search_runbooks Jaccard score above which a benign-pattern match is
# trusted enough to suppress a break outright. Deliberately conservative —
# false suppression (missing a real break) is worse than one extra
# DIAGNOSE call on something that turns out benign.
BENIGN_MATCH_THRESHOLD = 0.15
BENIGN_RUNBOOK_IDS = {"RB-014"}  # clock-skew runbook seeded in Stage 1


def _call(tool_fn, *args, **kwargs):
    """Unwrap either a plain function (mcp 1.x @tool()) or an object
    exposing .fn — keeps this module working across mcp SDK versions."""
    target = getattr(tool_fn, "fn", tool_fn)
    return target(*args, **kwargs)


def _txn_evidence(brk: ReconBreak) -> list[EvidenceItem]:
    items = []
    for system in (brk.source_system, brk.counter_system):
        rows = _call(ledger_telemetry.get_txn_record, brk.txn_ref, system)
        if rows:
            for row in rows:
                items.append(EvidenceItem(
                    evidence_id=f"{brk.break_id}-TXN-{system}-{len(items)}",
                    source_server="ledger-telemetry",
                    kind="txn_record",
                    timestamp=row.get("ts"),
                    content=f"{system}: txn_ref={row['txn_ref']} amount={row['amount']} "
                            f"{row['currency']} status={row['status']} ts={row['ts']}",
                ))
        else:
            # A MISSING record is itself evidence (e.g. timing_mismatch
            # pattern: settlement row doesn't exist yet).
            items.append(EvidenceItem(
                evidence_id=f"{brk.break_id}-TXN-{system}-missing",
                source_server="ledger-telemetry",
                kind="txn_record",
                timestamp=None,
                content=f"{system}: no record found for txn_ref={brk.txn_ref}",
            ))
    return items


def _batch_evidence(brk: ReconBreak) -> list[EvidenceItem]:
    start = (brk.detected_at - EVIDENCE_WINDOW).isoformat()
    end = (brk.detected_at + EVIDENCE_WINDOW).isoformat()
    rows = _call(ledger_telemetry.get_batch_status, start, end)
    return [
        EvidenceItem(
            evidence_id=f"{brk.break_id}-BATCH-{i}",
            source_server="ledger-telemetry",
            kind="batch_status",
            timestamp=row.get("started_at"),
            content=f"{row['batch_id']} ({row['job_name']}) started={row['started_at']} "
                    f"completed={row['completed_at']} status={row['status']}",
        )
        for i, row in enumerate(rows)
    ]


def _deploy_evidence(brk: ReconBreak) -> list[EvidenceItem]:
    start = (brk.detected_at - DEPLOY_LOOKBACK_WINDOW).isoformat()
    end = (brk.detected_at + EVIDENCE_WINDOW).isoformat()
    rows = _call(obs_context.get_deploy_and_schema_events, start, end)
    return [
        EvidenceItem(
            evidence_id=f"{brk.break_id}-DEPLOY-{i}",
            source_server="obs-context",
            kind="deploy_event",
            timestamp=row.get("ts"),
            content=f"{row['service']}: {row['kind']} — {row['description']} (at {row['ts']})",
        )
        for i, row in enumerate(rows)
    ]


def _log_evidence(brk: ReconBreak) -> list[EvidenceItem]:
    start = (brk.detected_at - EVIDENCE_WINDOW).isoformat()
    end = (brk.detected_at + EVIDENCE_WINDOW).isoformat()
    items = []
    # Only known service names get queried — extend this list as the
    # synthetic dataset grows more services.
    for service in ("settlement-service",):
        rows = _call(obs_context.get_logs, service, start, end)
        for i, row in enumerate(rows):
            items.append(EvidenceItem(
                evidence_id=f"{brk.break_id}-LOG-{service}-{i}",
                source_server="obs-context",
                kind="log_line",
                timestamp=row.get("ts"),
                content=f"[{row['level']}] {row['service']}: {row['message']}",
            ))
    return items


def _runbook_evidence(brk: ReconBreak) -> tuple[list[EvidenceItem], bool]:
    """Returns (evidence_items, is_benign_known_pattern)."""
    hits = _call(runbook_kb.search_runbooks, brk.description, top_k=3)
    items = [
        EvidenceItem(
            evidence_id=f"{brk.break_id}-RB-{hit['id']}",
            source_server="runbook-kb",
            kind="runbook_match",
            timestamp=None,
            content=f"{hit['id']} ({hit['title']}), score={hit['score']}: {hit['text']}",
        )
        for hit in hits
    ]

    # Past similar incidents (episodic memory) — empty until reflect_node
    # has actually persisted at least one outcome (Stage 6). Surfaced with
    # the same runbook_match kind since both are retrieval matches served
    # by runbook-kb; the content text makes clear which is which.
    past_hits = _call(runbook_kb.find_similar_past_breaks, brk.description, top_k=2)
    for hit in past_hits:
        items.append(EvidenceItem(
            evidence_id=f"{brk.break_id}-PAST-{hit['break_id']}",
            source_server="runbook-kb",
            kind="runbook_match",
            timestamp=hit.get("stored_at"),
            content=f"Similar past incident {hit['break_id']} (score={hit['score']}): "
                    f"cause={hit['final_cause']!r}, action={hit['action_taken']}, "
                    f"resolved={bool(hit['outcome_resolved'])}, lessons={hit['lessons']!r}",
        ))

    is_benign = any(
        hit["id"] in BENIGN_RUNBOOK_IDS and hit["score"] >= BENIGN_MATCH_THRESHOLD
        for hit in hits
    )
    return items, is_benign


def _check_duplicate(brk: ReconBreak) -> bool:
    """Crude duplicate-open-incident check: is there more than one break
    on the feed for this same txn_ref? (Real dedupe against in-flight
    graph threads / episodic memory comes online in Stage 6 — this is a
    reasonable stand-in given what ledger-telemetry can see today.)
    """
    all_breaks = _call(ledger_telemetry.get_new_breaks, "2000-01-01T00:00:00")
    matching = [b for b in all_breaks if b["txn_ref"] == brk.txn_ref]
    return len(matching) > 1


def gather_evidence(brk: ReconBreak) -> EvidenceBundle:
    items: list[EvidenceItem] = []
    items += _txn_evidence(brk)
    items += _batch_evidence(brk)
    items += _deploy_evidence(brk)
    items += _log_evidence(brk)

    runbook_items, is_benign = _runbook_evidence(brk)
    items += runbook_items

    is_duplicate = _check_duplicate(brk)

    return EvidenceBundle(
        break_id=brk.break_id,
        items=items,
        is_duplicate=is_duplicate,
        is_benign_known_pattern=is_benign,
    )
