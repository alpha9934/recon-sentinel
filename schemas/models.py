"""
Structured-output contracts for Recon Sentinel.

The DIAGNOSE node is the ONLY place an LLM call happens. Its output MUST
validate against DiagnosisResult below — nothing downstream trusts free text.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Break intake
# ---------------------------------------------------------------------------

class BreakSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReconBreak(BaseModel):
    """A single detected mismatch, as produced by MONITOR."""
    break_id: str
    detected_at: datetime
    source_system: Literal["core_ledger", "payment_gateway", "settlement"]
    counter_system: Literal["core_ledger", "payment_gateway", "settlement"]
    txn_ref: str
    amount_mismatch: Optional[float] = None
    description: str
    severity: BreakSeverity


# ---------------------------------------------------------------------------
# Evidence gathered by TRIAGE (deterministic MCP calls, no LLM)
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    evidence_id: str
    source_server: Literal["ledger-telemetry", "obs-context", "runbook-kb"]
    kind: Literal["txn_record", "batch_status", "log_line", "trace_span",
                  "deploy_event", "schema_change", "runbook_match"]
    timestamp: Optional[datetime] = None
    content: str = Field(..., description="Raw or summarized evidence text")
    ref_url: Optional[str] = None


class EvidenceBundle(BaseModel):
    break_id: str
    items: list[EvidenceItem]
    is_duplicate: bool = False
    is_benign_known_pattern: bool = False


# ---------------------------------------------------------------------------
# DIAGNOSE output — the ONE LLM-generated structured object in the system
# ---------------------------------------------------------------------------

class RootCauseHypothesis(BaseModel):
    rank: int = Field(..., ge=1)
    cause: str = Field(..., description="Concise root-cause statement")
    confidence: float = Field(..., ge=0.0, le=1.0)
    cited_evidence_ids: list[str] = Field(
        ..., min_length=1,
        description="evidence_id values this hypothesis is grounded in — "
                    "never allowed to be empty, enforced at schema level",
    )
    reasoning: str = Field(..., description="Short evidence-to-conclusion chain")


class DiagnosisResult(BaseModel):
    break_id: str
    hypotheses: list[RootCauseHypothesis] = Field(..., min_length=1, max_length=3)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    requires_escalation: bool

    @field_validator("hypotheses")
    @classmethod
    def ranks_are_sequential(cls, v: list[RootCauseHypothesis]):
        ranks = sorted(h.rank for h in v)
        if ranks != list(range(1, len(v) + 1)):
            raise ValueError("hypothesis ranks must be sequential starting at 1")
        return v


# ---------------------------------------------------------------------------
# PLAN ACTION / HUMAN APPROVAL / ACT
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    MARK_RESOLVED = "mark_resolved"
    RERUN_JOB = "rerun_job"
    FLAG_FOR_REVIEW = "flag_for_review"
    NO_ACTION = "no_action"


class ProposedAction(BaseModel):
    break_id: str
    action_type: ActionType
    target_ref: str
    justification: str
    based_on_hypothesis_rank: int


class ApprovalToken(BaseModel):
    """Single-use, scoped token minted only by a real external approval event.

    This is NOT something the LLM produces. It's minted by the approval
    service after a human clicks approve, and it's checked, then burned,
    by the ACT node before any recon-actuator call is made.
    """
    token_id: str
    break_id: str
    action_type: ActionType
    approved_by: str
    approved_at: datetime
    expires_at: datetime
    used: bool = False


class ApprovalDecision(BaseModel):
    break_id: str
    decision: Literal["approved", "rejected", "timeout"]
    token: Optional[ApprovalToken] = None
    reviewer_note: Optional[str] = None


# ---------------------------------------------------------------------------
# VERIFY / REFLECT
# ---------------------------------------------------------------------------

class VerificationResult(BaseModel):
    break_id: str
    action_taken: ActionType
    resolved: bool
    verification_evidence_ids: list[str]
    notes: str


class ReflectionRecord(BaseModel):
    """Written to episodic memory — past break -> outcome."""
    break_id: str
    final_hypothesis: RootCauseHypothesis
    action_taken: ActionType
    outcome_resolved: bool
    lessons: str
    stored_at: datetime
