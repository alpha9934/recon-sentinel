"""
recon-actuator MCP server — WRITE, APPROVAL-GATED.

This is the ONLY MCP server in the system with write credentials to
anything. In a real deployment it runs as a physically separate process
from the three read-only servers, with its own DB role (INSERT/UPDATE
only on the specific action tables it needs). In this synthetic/personal-
project setup everything lives in one SQLite file for simplicity, but the
process/module boundary is still real: this file is the only place in
the entire codebase that opens a WRITABLE connection to recon.db.

Every tool here REQUIRES a valid ApprovalToken: unexpired, unused, and
matching (break_id, action_type). The token is validated and burned
(marked used=True, persisted to the tokens table) inside this process —
it does not trust the caller's claim that approval happened.

Run: python -m mcp_servers.recon_actuator.server
"""
import sqlite3
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from schemas.models import ActionType, ApprovalToken

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "recon.db"

mcp = FastMCP("recon-actuator")

# Tables this server owns writes to. Created lazily (idempotent) so Stage 1's
# generator doesn't need to know about them — the write surface belongs to
# this server, not to data generation.
_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS used_approval_tokens (
    token_id TEXT PRIMARY KEY, break_id TEXT, action_type TEXT,
    approved_by TEXT, used_at TEXT
);
CREATE TABLE IF NOT EXISTS actions_taken (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    break_id TEXT, action_type TEXT, txn_ref TEXT,
    note TEXT, token_id TEXT, taken_at TEXT
);
CREATE TABLE IF NOT EXISTS review_queue (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    break_id TEXT, txn_ref TEXT, note TEXT, status TEXT, created_at TEXT
);
"""


def _rw_connection() -> sqlite3.Connection:
    """The only writable connection to recon.db anywhere in this codebase."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_MIGRATION_SQL)
    return conn


def _validate_and_burn_token(conn: sqlite3.Connection, token: ApprovalToken,
                              expected_break_id: str, expected_action: ActionType) -> None:
    """Checked against the DB, not just the in-memory object — a token
    already burned in a previous process/request must be rejected even
    if the caller (bug or otherwise) still has a Python object with
    used=False."""
    already_used = conn.execute(
        "SELECT 1 FROM used_approval_tokens WHERE token_id = ?", (token.token_id,)
    ).fetchone()
    if already_used or token.used:
        raise PermissionError(f"approval token {token.token_id} already used")
    if token.break_id != expected_break_id:
        raise PermissionError(
            f"token break_id mismatch: token is for {token.break_id}, "
            f"action is for {expected_break_id}"
        )
    if token.action_type != expected_action:
        raise PermissionError(
            f"token action_type mismatch: token approves {token.action_type}, "
            f"action being taken is {expected_action}"
        )
    expires_at = token.expires_at
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if datetime.utcnow() > expires_at:
        raise PermissionError(f"approval token {token.token_id} expired at {expires_at}")

    conn.execute(
        "INSERT INTO used_approval_tokens VALUES (?,?,?,?,?)",
        (token.token_id, token.break_id, token.action_type.value,
         token.approved_by, datetime.utcnow().isoformat()),
    )
    token.used = True


@mcp.tool()
def mark_resolved(break_id: str, txn_ref: str, token: ApprovalToken) -> dict:
    """Records the break as resolved without further data mutation —
    appropriate for patterns like currency_rounding_mismatch where the
    drift is within tolerance and no correction is actually needed,
    just a documented sign-off.
    """
    with _rw_connection() as conn:
        _validate_and_burn_token(conn, token, break_id, ActionType.MARK_RESOLVED)
        conn.execute(
            "INSERT INTO actions_taken (break_id, action_type, txn_ref, note, "
            "token_id, taken_at) VALUES (?,?,?,?,?,?)",
            (break_id, ActionType.MARK_RESOLVED.value, txn_ref,
             "Marked resolved — drift within tolerance, no correction needed",
             token.token_id, datetime.utcnow().isoformat()),
        )
        conn.commit()
    return {"status": "resolved", "break_id": break_id, "txn_ref": txn_ref}


@mcp.tool()
def rerun_job(break_id: str, txn_ref: str, token: ApprovalToken) -> dict:
    """Simulates re-running the settlement batch for one transaction —
    the correct remediation for timing_mismatch, where the settlement
    batch ran before the upstream feed had settled and simply needs to
    process the transaction it originally missed.

    Concretely: copies the core_ledger record for txn_ref into
    settlement if it's not already there, exactly what a real batch
    re-run would produce for a transaction that only missed its window.
    """
    with _rw_connection() as conn:
        _validate_and_burn_token(conn, token, break_id, ActionType.RERUN_JOB)

        existing = conn.execute(
            "SELECT 1 FROM settlement WHERE txn_ref = ?", (txn_ref,)
        ).fetchone()
        if not existing:
            ledger_row = conn.execute(
                "SELECT * FROM core_ledger WHERE txn_ref = ?", (txn_ref,)
            ).fetchone()
            if ledger_row:
                conn.execute(
                    "INSERT INTO settlement VALUES (?,?,?,?,?)",
                    (ledger_row["txn_ref"], ledger_row["amount"],
                     ledger_row["currency"], datetime.utcnow().isoformat(), "posted"),
                )

        conn.execute(
            "INSERT INTO actions_taken (break_id, action_type, txn_ref, note, "
            "token_id, taken_at) VALUES (?,?,?,?,?,?)",
            (break_id, ActionType.RERUN_JOB.value, txn_ref,
             "Re-ran settlement batch for missed transaction",
             token.token_id, datetime.utcnow().isoformat()),
        )
        conn.commit()
    return {"status": "rerun_complete", "break_id": break_id, "txn_ref": txn_ref}


@mcp.tool()
def flag_for_review(break_id: str, txn_ref: str, note: str,
                     token: ApprovalToken) -> dict:
    """Escalates to a human review queue — appropriate for patterns
    (schema_break, duplicate_submission) where the correct fix requires
    engineering judgment or a reversal decision this system should never
    make autonomously, even with approval to *flag* it.
    """
    with _rw_connection() as conn:
        _validate_and_burn_token(conn, token, break_id, ActionType.FLAG_FOR_REVIEW)
        conn.execute(
            "INSERT INTO review_queue (break_id, txn_ref, note, status, created_at) "
            "VALUES (?,?,?,?,?)",
            (break_id, txn_ref, note, "open", datetime.utcnow().isoformat()),
        )
        conn.execute(
            "INSERT INTO actions_taken (break_id, action_type, txn_ref, note, "
            "token_id, taken_at) VALUES (?,?,?,?,?,?)",
            (break_id, ActionType.FLAG_FOR_REVIEW.value, txn_ref, note,
             token.token_id, datetime.utcnow().isoformat()),
        )
        conn.commit()
    return {"status": "flagged", "break_id": break_id, "txn_ref": txn_ref}


if __name__ == "__main__":
    mcp.run()
