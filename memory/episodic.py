"""
Episodic memory: past break -> outcome (ReflectionRecord objects).

Kept strictly separate from:
  - working state (per-incident ReconState, lives only for the graph run)
  - semantic memory (the runbook corpus itself, in runbook-kb/runbooks.json)

This separation matters for the interview talking point: the system
doesn't let "what happened in one incident" leak into "general domain
knowledge" without going through an explicit promotion step (e.g. a
human curator turning a recurring reflection into a written runbook).
Nothing here writes to runbooks.json, and nothing in runbooks.json is
mutated by what gets learned episodically.

Backing store: a table in the same synthetic recon.db SQLite file (a
real deployment would use a proper vector store here — see the
"upgrade path" note in mcp_servers/runbook_kb/server.py for the same
pattern already used for search_runbooks).
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

from schemas.models import ReflectionRecord

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "recon.db"

_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS episodic_memory (
    break_id TEXT PRIMARY KEY,
    break_signature TEXT,
    final_cause TEXT,
    action_taken TEXT,
    outcome_resolved INTEGER,
    lessons TEXT,
    stored_at TEXT
);
"""

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "of",
    "in", "on", "for", "with", "if", "be", "this", "that", "it", "as", "by",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _rw_connection() -> sqlite3.Connection:
    """Episodic memory needs to write (reflect_node persisting outcomes)
    as well as read (find_similar_past_breaks, find_open_incident) — this
    is the second and only other place in the codebase besides
    recon-actuator that opens a writable connection to recon.db, and it's
    deliberately scoped to just this one table via the migration above."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_MIGRATION_SQL)
    return conn


def write(record: ReflectionRecord, break_signature: str) -> None:
    """Persist an outcome. Upserts on break_id — reflecting on the same
    break twice (e.g. a retry) updates rather than duplicates."""
    with _rw_connection() as conn:
        conn.execute(
            "INSERT INTO episodic_memory "
            "(break_id, break_signature, final_cause, action_taken, "
            "outcome_resolved, lessons, stored_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(break_id) DO UPDATE SET "
            "break_signature=excluded.break_signature, "
            "final_cause=excluded.final_cause, "
            "action_taken=excluded.action_taken, "
            "outcome_resolved=excluded.outcome_resolved, "
            "lessons=excluded.lessons, stored_at=excluded.stored_at",
            (
                record.break_id, break_signature, record.final_hypothesis.cause,
                record.action_taken.value, int(record.outcome_resolved),
                record.lessons, record.stored_at.isoformat(),
            ),
        )
        conn.commit()


def find_open_incident(break_signature: str) -> dict | None:
    """Used by TRIAGE's is_duplicate check: has an incident with this
    exact signature already been reflected on and NOT resolved? (An
    already-resolved past incident with the same signature isn't a
    duplicate-in-flight, it's just a recurring pattern — that's useful
    context for DIAGNOSE via find_similar_past_breaks, not a suppress
    signal.)
    """
    with _rw_connection() as conn:
        row = conn.execute(
            "SELECT * FROM episodic_memory WHERE break_signature = ? "
            "AND outcome_resolved = 0",
            (break_signature,),
        ).fetchone()
        return dict(row) if row else None


def search_similar(query_text: str, top_k: int = 5) -> list[dict]:
    """Keyword-overlap search over past incidents, same Jaccard-similarity
    approach as runbook_kb.search_runbooks for consistency — see that
    module's docstring for the embedding-based upgrade path this would
    also take in a real deployment.
    """
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    with _rw_connection() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM episodic_memory").fetchall()]

    scored = []
    for row in rows:
        doc_text = f"{row['break_signature']} {row['final_cause']} {row['lessons']}"
        doc_tokens = _tokenize(doc_text)
        overlap = query_tokens & doc_tokens
        union = query_tokens | doc_tokens
        score = len(overlap) / len(union) if union else 0.0
        if score > 0:
            scored.append({**row, "score": round(score, 4)})

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]
