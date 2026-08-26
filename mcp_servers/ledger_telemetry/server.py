"""
ledger-telemetry MCP server — READ ONLY.

Runs as its own process, its own credentials (a DB role with SELECT-only
grants on the synthetic ledger/gateway/settlement tables). This process
has no code path that can issue a write — that's the structural guarantee,
not a convention someone could accidentally break in a shared client.

Backed by the synthetic SQLite DB from data/synthetic/generate.py. Every
tool opens a fresh read-only connection (sqlite3 URI mode=ro) — belt and
suspenders alongside the "separate process, separate credentials" story:
even a bug in this server's own code can't accidentally write, because
the connection itself refuses writes at the driver level.

Run: python -m mcp_servers.ledger_telemetry.server
"""
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "recon.db"

mcp = FastMCP("ledger-telemetry")


def _ro_connection() -> sqlite3.Connection:
    """Read-only connection — sqlite3 enforces this at the driver level,
    independent of whatever DB role/grants a real Postgres deployment
    would layer on top."""
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def get_new_breaks(since_iso: str) -> list[dict]:
    """Return newly detected reconciliation breaks since a timestamp.

    Reads from break_feed (populated by seeded incidents in
    data/synthetic/generate.py). In a real deployment this table would be
    fed by an actual break-detection job; here it's pre-seeded.
    """
    with _ro_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM break_feed WHERE detected_at >= ? ORDER BY detected_at",
            (since_iso,),
        ).fetchall()
        return [dict(r) for r in rows]


@mcp.tool()
def get_txn_record(txn_ref: str, system: str) -> dict | None:
    """Fetch a single transaction record from core_ledger / payment_gateway
    / settlement (read-only synthetic tables).

    system must be one of: core_ledger, payment_gateway, settlement.
    Returns None if no record exists in that system for that txn_ref —
    a missing record IS evidence (e.g. the timing-mismatch pattern).
    """
    if system not in ("core_ledger", "payment_gateway", "settlement"):
        raise ValueError(f"unknown system: {system}")
    with _ro_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {system} WHERE txn_ref = ?", (txn_ref,)  # noqa: S608
            # system is allowlist-checked above, not user-composed SQL
        ).fetchall()
        return [dict(r) for r in rows]


@mcp.tool()
def get_batch_status(batch_window_start: str, batch_window_end: str) -> list[dict]:
    """Fetch batch job run statuses in a time window (for timing-mismatch
    diagnosis, e.g. 'settlement batch ran before gateway batch completed')."""
    with _ro_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM batch_jobs WHERE started_at BETWEEN ? AND ? "
            "ORDER BY started_at",
            (batch_window_start, batch_window_end),
        ).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    mcp.run()
