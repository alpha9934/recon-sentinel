"""
runbook-kb MCP server — READ ONLY.

Retrieval over (a) written runbooks for known break patterns and
(b) episodic memory of past break -> outcome (semantic + episodic tiers).

Stage 2 note: search_runbooks below uses simple keyword-overlap scoring
against data/synthetic/runbooks.json, NOT real embeddings — this is
intentionally the cheapest thing that lets triage_node run end-to-end.
Swap in a FAISS/Pinecone index over sentence embeddings before relying on
this for anything beyond exact/near-exact phrase matches (TODO marked
below). find_similar_past_breaks stays a stub until Stage 6 (episodic
memory) exists — there's nothing to search yet.

Run: python -m mcp_servers.runbook_kb.server
"""
import json
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

RUNBOOKS_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "runbooks.json"

mcp = FastMCP("runbook-kb")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "of",
    "in", "on", "for", "with", "if", "be", "this", "that", "it", "as", "by",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _load_runbooks() -> list[dict]:
    with open(RUNBOOKS_PATH) as f:
        return json.load(f)


@mcp.tool()
def search_runbooks(query: str, top_k: int = 5) -> list[dict]:
    """Keyword-overlap search over the runbook corpus, ranked by Jaccard
    similarity between query tokens and (title + text) tokens.

    TODO (upgrade path): replace with embedding similarity — embed
    runbooks.json once at startup into a FAISS index, embed the query,
    cosine-rank. Keeping the tool signature identical means nothing
    calling this tool needs to change when you make that swap.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored = []
    for rb in _load_runbooks():
        doc_tokens = _tokenize(rb["title"] + " " + rb["text"])
        overlap = query_tokens & doc_tokens
        union = query_tokens | doc_tokens
        score = len(overlap) / len(union) if union else 0.0
        if score > 0:
            scored.append({**rb, "score": round(score, 4)})

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


@mcp.tool()
def find_similar_past_breaks(break_signature: str, top_k: int = 5) -> list[dict]:
    """Nearest-neighbor search over episodic memory of resolved breaks —
    used both for TRIAGE's benign-pattern check and to enrich DIAGNOSE.

    Delegates to memory/episodic.py's search_similar, which uses the same
    keyword-overlap approach as search_runbooks above (see that function's
    docstring for the embedding-based upgrade path). Returns an empty list
    until reflect_node has actually persisted at least one outcome — that's
    expected on a fresh database, not an error.
    """
    from memory import episodic
    return episodic.search_similar(break_signature, top_k=top_k)


if __name__ == "__main__":
    mcp.run()
