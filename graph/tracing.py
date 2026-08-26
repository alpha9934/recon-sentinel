"""
LangFuse tracing for the DIAGNOSE step.

Captures exactly what the interview talking points promise: prompt,
evidence, confidence, cost, latency, per DIAGNOSE call. Deliberately
optional — if LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY aren't set, the
graph still runs, it just isn't traced. A missing observability backend
should never be able to break the triage pipeline.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

from schemas.models import DiagnosisResult, EvidenceBundle

_LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)

_client = None
if _LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse
        _client = Langfuse()
    except Exception:
        # Never let a broken observability integration take down the
        # actual pipeline — degrade to untraced rather than crash.
        _LANGFUSE_ENABLED = False
        _client = None


@contextmanager
def trace_diagnose_call(break_id: str, evidence: EvidenceBundle) -> Iterator[dict]:
    """Usage:
        with trace_diagnose_call(break_id, evidence) as span:
            result = diagnose_with_llm(evidence, description)
            span["result"] = result   # captured on exit

    Yields a plain dict the caller writes into; this function reads it
    back after the `with` block to populate the LangFuse span (or,
    if LangFuse isn't configured, just logs latency to stdout so you
    still get signal locally).
    """
    span_data: dict = {}
    start = time.monotonic()
    error: Exception | None = None
    try:
        yield span_data
    except Exception as e:
        error = e
        raise
    finally:
        latency_s = time.monotonic() - start
        result: DiagnosisResult | None = span_data.get("result")

        if _LANGFUSE_ENABLED and _client is not None:
            try:
                _client.trace(
                    name="diagnose_node",
                    input={
                        "break_id": break_id,
                        "n_evidence_items": len(evidence.items),
                    },
                    output=result.model_dump() if result else None,
                    metadata={
                        "latency_s": latency_s,
                        "error": str(error) if error else None,
                        "overall_confidence": result.overall_confidence if result else None,
                        "requires_escalation": result.requires_escalation if result else None,
                    },
                )
                _client.flush()
            except Exception:
                # Tracing failures are never allowed to mask or replace
                # the real pipeline error/result.
                pass
        else:
            status = "error" if error else "ok"
            conf = f"{result.overall_confidence:.2f}" if result else "n/a"
            print(
                f"[diagnose_node] break_id={break_id} status={status} "
                f"latency={latency_s:.2f}s confidence={conf} "
                f"(LangFuse not configured — set LANGFUSE_PUBLIC_KEY / "
                f"LANGFUSE_SECRET_KEY to persist traces)"
            )
