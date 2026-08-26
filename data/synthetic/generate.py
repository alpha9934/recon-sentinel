"""
Synthetic data generator for Recon Sentinel.

Produces:
  - data/synthetic/recon.db (SQLite) with tables:
      core_ledger, payment_gateway, settlement   (the three systems)
      batch_jobs                                 (for timing-mismatch evidence)
      logs, traces, deploy_events                (obs-context evidence)
      break_feed                                 (what ledger-telemetry's
                                                   get_new_breaks() reads from)
  - data/synthetic/golden_incidents.jsonl — one JSON line per seeded break,
    with break_event + evidence + true_cause + true_action, matching the
    Pydantic schemas in schemas/models.py.

Deterministic (fixed seed) so re-running regenerates identical data —
important for reproducible eval baselines.

Usage:
    python -m data.synthetic.generate
"""
from __future__ import annotations

import json
import random
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
OUT_DIR = Path(__file__).parent
DB_PATH = OUT_DIR / "recon.db"
GOLDEN_PATH = OUT_DIR / "golden_incidents.jsonl"

N_CLEAN_TXNS = 300
N_BREAKS_PER_PATTERN = 12
BASE_TIME = datetime(2026, 8, 1, 0, 0, 0)

random.seed(SEED)


# ---------------------------------------------------------------------------
# Break pattern definitions — each maps to a known true_cause / true_action,
# and to a specific "shape" of corruption + supporting evidence to seed.
# ---------------------------------------------------------------------------

BREAK_PATTERNS = [
    "timing_mismatch",
    "schema_break",
    "duplicate_submission",
    "currency_rounding_mismatch",
    "benign_clock_skew",
]

TRUE_CAUSE = {
    "timing_mismatch": "settlement batch ran before gateway feed settled",
    "schema_break": "schema change on settlement side broke field mapping",
    "duplicate_submission": "duplicate transaction submitted to payment gateway",
    "currency_rounding_mismatch": "currency conversion rounding mismatch between gateway and settlement",
    "benign_clock_skew": "expected clock skew between systems, self-resolves within reconciliation window",
}

TRUE_ACTION = {
    "timing_mismatch": "rerun_job",
    "schema_break": "flag_for_review",
    "duplicate_submission": "flag_for_review",
    "currency_rounding_mismatch": "mark_resolved",
    "benign_clock_skew": "no_action",
}


@dataclass
class Txn:
    txn_ref: str
    amount: float
    currency: str
    timestamp: datetime


def rand_time(offset_days_max=20) -> datetime:
    return BASE_TIME + timedelta(
        days=random.randint(0, offset_days_max),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )


def new_txn() -> Txn:
    return Txn(
        txn_ref=f"TXN-{random.randint(10000, 99999)}",
        amount=round(random.uniform(10, 5000), 2),
        currency=random.choice(["INR", "USD", "EUR"]),
        timestamp=rand_time(),
    )


# ---------------------------------------------------------------------------
# SQLite schema + population
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE core_ledger (
    txn_ref TEXT, amount REAL, currency TEXT, ts TEXT, status TEXT
);
CREATE TABLE payment_gateway (
    txn_ref TEXT, amount REAL, currency TEXT, ts TEXT, status TEXT
);
CREATE TABLE settlement (
    txn_ref TEXT, amount REAL, currency TEXT, ts TEXT, status TEXT
);
CREATE TABLE batch_jobs (
    batch_id TEXT, job_name TEXT, started_at TEXT, completed_at TEXT, status TEXT
);
CREATE TABLE logs (
    log_id TEXT, service TEXT, ts TEXT, level TEXT, message TEXT
);
CREATE TABLE traces (
    trace_id TEXT, txn_ref TEXT, span_name TEXT, ts TEXT, duration_ms INTEGER
);
CREATE TABLE deploy_events (
    event_id TEXT, service TEXT, ts TEXT, kind TEXT, description TEXT
);
CREATE TABLE break_feed (
    break_id TEXT, detected_at TEXT, source_system TEXT, counter_system TEXT,
    txn_ref TEXT, amount_mismatch REAL, description TEXT, severity TEXT
);
"""


def build_database(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)

    # --- clean, fully matched transactions across all three systems ---
    for _ in range(N_CLEAN_TXNS):
        t = new_txn()
        ts = t.timestamp.isoformat()
        for table in ("core_ledger", "payment_gateway", "settlement"):
            conn.execute(
                f"INSERT INTO {table} VALUES (?,?,?,?,?)",
                (t.txn_ref, t.amount, t.currency, ts, "matched"),
            )

    conn.commit()


def seed_timing_mismatch(conn, i: int) -> dict:
    t = new_txn()
    break_id = f"BRK-TM-{i:03d}"
    gateway_settle_time = t.timestamp
    batch_start = gateway_settle_time - timedelta(minutes=5)  # ran too early
    batch_id = f"BATCH-{uuid.uuid4().hex[:8]}"

    conn.execute("INSERT INTO core_ledger VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount, t.currency, t.timestamp.isoformat(), "posted"))
    conn.execute("INSERT INTO payment_gateway VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount, t.currency, gateway_settle_time.isoformat(), "settled"))
    # settlement row is MISSING — the break — because the batch ran first
    conn.execute("INSERT INTO batch_jobs VALUES (?,?,?,?,?)",
                 (batch_id, "settlement_batch_04", batch_start.isoformat(),
                  (batch_start + timedelta(minutes=2)).isoformat(), "completed"))

    conn.execute("INSERT INTO break_feed VALUES (?,?,?,?,?,?,?,?)",
                 (break_id, t.timestamp.isoformat(), "core_ledger", "settlement",
                  t.txn_ref, t.amount, "Ledger posted, no matching settlement record", "high"))

    evidence = [
        {"evidence_id": f"{break_id}-EV1", "source_server": "ledger-telemetry",
         "kind": "batch_status", "timestamp": batch_start.isoformat(),
         "content": f"{batch_id} (settlement_batch_04) started at {batch_start.isoformat()}, "
                     f"BEFORE gateway settlement at {gateway_settle_time.isoformat()}"},
        {"evidence_id": f"{break_id}-EV2", "source_server": "ledger-telemetry",
         "kind": "txn_record", "timestamp": t.timestamp.isoformat(),
         "content": f"{t.txn_ref} posted in core_ledger, status=posted, amount={t.amount} {t.currency}"},
    ]
    return _incident(break_id, t, "core_ledger", "settlement", evidence, "timing_mismatch", "high")


def seed_schema_break(conn, i: int) -> dict:
    t = new_txn()
    break_id = f"BRK-SB-{i:03d}"
    deploy_time = t.timestamp - timedelta(hours=1)
    event_id = f"DEPLOY-{uuid.uuid4().hex[:8]}"

    conn.execute("INSERT INTO core_ledger VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount, t.currency, t.timestamp.isoformat(), "posted"))
    # settlement amount is wrong (0.0) because a field mapping broke post-deploy
    conn.execute("INSERT INTO settlement VALUES (?,?,?,?,?)",
                 (t.txn_ref, 0.0, t.currency, t.timestamp.isoformat(), "posted"))
    conn.execute("INSERT INTO deploy_events VALUES (?,?,?,?,?)",
                 (event_id, "settlement-service", deploy_time.isoformat(), "schema_change",
                  "Renamed field 'txn_amount' -> 'amount_cents' in settlement ingestion schema"))
    conn.execute("INSERT INTO logs VALUES (?,?,?,?,?)",
                 (f"LOG-{uuid.uuid4().hex[:8]}", "settlement-service", t.timestamp.isoformat(),
                  "ERROR", f"KeyError: 'txn_amount' not found while mapping {t.txn_ref}"))

    conn.execute("INSERT INTO break_feed VALUES (?,?,?,?,?,?,?,?)",
                 (break_id, t.timestamp.isoformat(), "core_ledger", "settlement",
                  t.txn_ref, t.amount, "Settlement amount recorded as 0.0", "high"))

    evidence = [
        {"evidence_id": f"{break_id}-EV1", "source_server": "obs-context",
         "kind": "deploy_event", "timestamp": deploy_time.isoformat(),
         "content": f"schema_change on settlement-service: renamed field 'txn_amount' -> "
                     f"'amount_cents', deployed {deploy_time.isoformat()}"},
        {"evidence_id": f"{break_id}-EV2", "source_server": "obs-context",
         "kind": "log_line", "timestamp": t.timestamp.isoformat(),
         "content": f"ERROR settlement-service: KeyError 'txn_amount' not found while mapping {t.txn_ref}"},
    ]
    return _incident(break_id, t, "core_ledger", "settlement", evidence, "schema_break", "high")


def seed_duplicate_submission(conn, i: int) -> dict:
    t = new_txn()
    break_id = f"BRK-DUP-{i:03d}"
    dup_time = t.timestamp + timedelta(seconds=30)

    conn.execute("INSERT INTO core_ledger VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount, t.currency, t.timestamp.isoformat(), "posted"))
    # gateway has TWO submissions for the same txn_ref
    conn.execute("INSERT INTO payment_gateway VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount, t.currency, t.timestamp.isoformat(), "settled"))
    conn.execute("INSERT INTO payment_gateway VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount, t.currency, dup_time.isoformat(), "settled"))
    conn.execute("INSERT INTO settlement VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount * 2, t.currency, t.timestamp.isoformat(), "posted"))

    conn.execute("INSERT INTO break_feed VALUES (?,?,?,?,?,?,?,?)",
                 (break_id, t.timestamp.isoformat(), "settlement", "core_ledger",
                  t.txn_ref, t.amount, "Settlement amount is 2x ledger amount", "critical"))

    evidence = [
        {"evidence_id": f"{break_id}-EV1", "source_server": "ledger-telemetry",
         "kind": "txn_record", "timestamp": dup_time.isoformat(),
         "content": f"payment_gateway has two settled entries for {t.txn_ref}: "
                     f"{t.timestamp.isoformat()} and {dup_time.isoformat()}, both amount={t.amount}"},
    ]
    return _incident(break_id, t, "settlement", "core_ledger", evidence,
                      "duplicate_submission", "critical")


def seed_currency_rounding(conn, i: int) -> dict:
    t = new_txn()
    break_id = f"BRK-FX-{i:03d}"
    rounded = round(t.amount * 1.0, 1)  # small rounding drift, e.g. paise-level
    drift = round(t.amount - rounded, 4)

    conn.execute("INSERT INTO payment_gateway VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount, t.currency, t.timestamp.isoformat(), "settled"))
    conn.execute("INSERT INTO settlement VALUES (?,?,?,?,?)",
                 (t.txn_ref, rounded, t.currency, t.timestamp.isoformat(), "posted"))

    conn.execute("INSERT INTO break_feed VALUES (?,?,?,?,?,?,?,?)",
                 (break_id, t.timestamp.isoformat(), "payment_gateway", "settlement",
                  t.txn_ref, drift, "Small amount mismatch, sub-currency-unit drift", "low"))

    evidence = [
        {"evidence_id": f"{break_id}-EV1", "source_server": "ledger-telemetry",
         "kind": "txn_record", "timestamp": t.timestamp.isoformat(),
         "content": f"gateway amount={t.amount} {t.currency}, settlement amount={rounded} "
                     f"{t.currency}, drift={drift} — consistent with rounding-mode mismatch"},
    ]
    return _incident(break_id, t, "payment_gateway", "settlement", evidence,
                      "currency_rounding_mismatch", "low")


def seed_benign_clock_skew(conn, i: int) -> dict:
    t = new_txn()
    break_id = f"BRK-CS-{i:03d}"
    skewed_ts = t.timestamp + timedelta(seconds=random.randint(1, 4))

    conn.execute("INSERT INTO core_ledger VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount, t.currency, t.timestamp.isoformat(), "posted"))
    conn.execute("INSERT INTO settlement VALUES (?,?,?,?,?)",
                 (t.txn_ref, t.amount, t.currency, skewed_ts.isoformat(), "posted"))

    conn.execute("INSERT INTO break_feed VALUES (?,?,?,?,?,?,?,?)",
                 (break_id, t.timestamp.isoformat(), "core_ledger", "settlement",
                  t.txn_ref, 0.0, "Timestamp delta within known NTP skew tolerance", "low"))

    evidence = [
        {"evidence_id": f"{break_id}-EV1", "source_server": "runbook-kb",
         "kind": "runbook_match", "timestamp": None,
         "content": "Matches known runbook pattern RB-014: sub-5-second clock skew "
                     "between core_ledger and settlement hosts is expected and self-resolves"},
    ]
    inc = _incident(break_id, t, "core_ledger", "settlement", evidence,
                     "benign_clock_skew", "low")
    inc["evidence"]["is_benign_known_pattern"] = True
    return inc


def _incident(break_id, t: Txn, source, counter, evidence, pattern, severity) -> dict:
    return {
        "break_event": {
            "break_id": break_id,
            "detected_at": t.timestamp.isoformat(),
            "source_system": source,
            "counter_system": counter,
            "txn_ref": t.txn_ref,
            "amount_mismatch": t.amount,
            "description": TRUE_CAUSE[pattern],
            "severity": severity,
        },
        "evidence": {
            "break_id": break_id,
            "items": evidence,
            "is_duplicate": False,
            "is_benign_known_pattern": False,
        },
        "true_cause": TRUE_CAUSE[pattern],
        "true_action": TRUE_ACTION[pattern],
        "pattern": pattern,
    }


SEED_FUNCS = {
    "timing_mismatch": seed_timing_mismatch,
    "schema_break": seed_schema_break,
    "duplicate_submission": seed_duplicate_submission,
    "currency_rounding_mismatch": seed_currency_rounding,
    "benign_clock_skew": seed_benign_clock_skew,
}


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    build_database(conn)

    incidents = []
    for pattern, fn in SEED_FUNCS.items():
        for i in range(1, N_BREAKS_PER_PATTERN + 1):
            incidents.append(fn(conn, i))
    conn.commit()
    conn.close()

    random.shuffle(incidents)
    with open(GOLDEN_PATH, "w") as f:
        for inc in incidents:
            f.write(json.dumps(inc) + "\n")

    print(f"Wrote {N_CLEAN_TXNS} clean txns + {len(incidents)} seeded breaks to {DB_PATH}")
    print(f"Wrote golden dataset ({len(incidents)} incidents) to {GOLDEN_PATH}")
    print(f"Pattern breakdown: {N_BREAKS_PER_PATTERN} each of {list(SEED_FUNCS.keys())}")


if __name__ == "__main__":
    main()
