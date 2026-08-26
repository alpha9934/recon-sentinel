"""
obs-context MCP server — READ ONLY.

Wraps read access to logs/traces/deploy-and-schema-change events
(a synthetic stand-in for Loki/Tempo + a deploy-events table), backed by
the same synthetic SQLite DB as ledger-telemetry — but note this is a
SEPARATE process with its own connection, mirroring how a real deployment
would have obs-context talk to an entirely different backend (observability
stack) than ledger-telemetry (core banking DB). They only share a file
here because both are synthetic stand-ins in one repo.

Run: python -m mcp_servers.obs_context.server
"""
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "recon.db"

mcp = FastMCP("obs-context")


def _ro_connection() -> sqlite3.Connection:
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def get_logs(service: str, start_iso: str, end_iso: str, query: str = "") -> list[dict]:
    """Fetch log lines for a service in a time window, optionally filtered
    by substring match against the log message."""
    sql = "SELECT * FROM logs WHERE service = ? AND ts BETWEEN ? AND ?"
    params: list = [service, start_iso, end_iso]
    if query:
        sql += " AND message LIKE ?"
        params.append(f"%{query}%")
    sql += " ORDER BY ts"
    with _ro_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


@mcp.tool()
def get_traces(txn_ref: str) -> list[dict]:
    """Fetch distributed trace spans correlated to a transaction reference.

    Note: the generator (Stage 1) doesn't currently seed the `traces`
    table with data — it's schema-ready but empty. Extend
    data/synthetic/generate.py if a break pattern needs trace evidence.
    """
    with _ro_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM traces WHERE txn_ref = ? ORDER BY ts", (txn_ref,)
        ).fetchall()
        return [dict(r) for r in rows]


@mcp.tool()
def get_deploy_and_schema_events(start_iso: str, end_iso: str) -> list[dict]:
    """Fetch deploy/schema-change events in a window — a huge share of
    reconciliation breaks trace back to one of these (see the
    schema_break pattern in data/synthetic/generate.py)."""
    with _ro_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM deploy_events WHERE ts BETWEEN ? AND ? ORDER BY ts",
            (start_iso, end_iso),
        ).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    mcp.run()
