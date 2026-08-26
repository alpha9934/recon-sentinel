"""
The single state object threaded through every node in the graph.

Kept intentionally flat and fully typed — this IS the audit record.
Every node reads a slice of it and writes back a slice of it; nothing
lives in hidden closures or module globals.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from schemas.models import (
    ApprovalDecision,
    DiagnosisResult,
    EvidenceBundle,
    ProposedAction,
    ReconBreak,
    VerificationResult,
)

GraphStatus = Literal[
    "monitoring",
    "triaging",
    "diagnosing",
    "planning",
    "awaiting_approval",
    "acting",
    "verifying",
    "reflecting",
    "suppressed",
    "escalated",
    "resolved",
]


class ReconState(TypedDict, total=False):
    # MONITOR
    break_event: ReconBreak

    # TRIAGE
    evidence: EvidenceBundle

    # DIAGNOSE (the one LLM call)
    diagnosis: DiagnosisResult

    # PLAN ACTION
    proposed_action: ProposedAction

    # HUMAN APPROVAL (interrupt)
    approval: ApprovalDecision

    # ACT / VERIFY
    verification: VerificationResult

    # control
    status: GraphStatus
    escalation_reason: Optional[str]
    retry_count: int
