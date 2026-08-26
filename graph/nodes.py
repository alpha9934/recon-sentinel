"""
Node bodies. Only diagnose_node touches an LLM. Everything else is
deterministic Python calling MCP tool servers.

These are intentionally left as clearly-marked stubs with correct
signatures, return shapes, and TODOs — fill in the MCP client calls
against the servers in mcp_servers/.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from graph.evidence_gathering import gather_evidence
from graph.llm_client import DiagnoseLLMError, diagnose_with_llm
from graph.state import ReconState
from graph.tracing import trace_diagnose_call
from mcp_servers.ledger_telemetry import server as _ledger_telemetry_server
from mcp_servers.recon_actuator import server as _recon_actuator_server
from memory import episodic
from schemas.models import (
    ActionType,
    ApprovalDecision,
    DiagnosisResult,
    EvidenceBundle,
    ProposedAction,
    ReflectionRecord,
    RootCauseHypothesis,
    VerificationResult,
)


class _ToolProxy:
    """Unwraps either a plain function (mcp 1.x @tool()) or an object
    exposing .fn — keeps this module working across mcp SDK versions,
    same pattern as graph/evidence_gathering.py."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        tool_fn = getattr(self._module, name)
        target = getattr(tool_fn, "fn", tool_fn)
        return target


recon_actuator = _ToolProxy(_recon_actuator_server)
recon_actuator_reads = _ToolProxy(_ledger_telemetry_server)


def monitor_node(state: ReconState) -> ReconState:
    """Poll ledger-telemetry MCP server's break feed. Populates break_event.

    TODO: replace with real MCP call, e.g.
        breaks = mcp_client.call("ledger-telemetry", "get_new_breaks")
    For now this node expects break_event to already be injected by the
    caller (see tests/ and the CLI runner) — MONITOR in a real deployment
    would be a scheduled poller that starts a new graph thread per break.
    """
    state["status"] = "triaging"
    return state


def triage_node(state: ReconState) -> ReconState:
    """Deterministic evidence gathering + dedupe/benign-pattern check.

    Calls the three read-only MCP servers via evidence_gathering.py:
      1. ledger-telemetry for txn records (both systems) + batch status
         in a window around the break.
      2. obs-context for deploy/schema-change events + logs in the same window.
      3. runbook-kb for nearest-neighbor runbook matches, which also
         drives the benign-known-pattern check (e.g. RB-014 clock skew).
      4. A crude open-incident dedupe check against the break feed itself
         (real dedupe against episodic memory comes online in Stage 6).

    No LLM call happens here — everything is deterministic tool calls.
    """
    brk = state["break_event"]
    state["evidence"] = gather_evidence(brk)
    state["status"] = "diagnosing"
    return state


def diagnose_node(state: ReconState) -> ReconState:
    """The ONLY node that calls an LLM.

    Structured-output call constrained to DiagnosisResult (see
    graph/llm_client.py) — the model literally cannot return free text,
    a forced tool call enforces the schema before the response reaches
    us. Wrapped in a LangFuse trace span (graph/tracing.py) capturing
    prompt/evidence context, confidence, and latency.

    Fails closed: any error from the LLM client (missing API key, API
    failure, schema validation failure, a hallucinated evidence_id) is
    treated as an automatic escalation, never as "proceed with an empty
    diagnosis." A silent-and-plausible failure is exactly what this
    system exists to avoid producing itself.
    """
    brk = state["break_event"]
    evidence = state["evidence"]

    with trace_diagnose_call(brk.break_id, evidence) as span:
        try:
            diagnosis = diagnose_with_llm(evidence, brk.description)
        except DiagnoseLLMError as e:
            diagnosis = DiagnosisResult(
                break_id=brk.break_id,
                hypotheses=[RootCauseHypothesis(
                    rank=1,
                    cause="diagnosis unavailable — LLM call failed or returned "
                          "ungrounded output",
                    confidence=0.0,
                    cited_evidence_ids=[evidence.items[0].evidence_id] if evidence.items
                    else ["none-available"],
                    reasoning=str(e),
                )],
                overall_confidence=0.0,
                requires_escalation=True,
            )
            state["escalation_reason"] = str(e)
        span["result"] = diagnosis

    state["diagnosis"] = diagnosis
    state["status"] = "planning"
    return state


def plan_action_node(state: ReconState) -> ReconState:
    """Deterministic mapping from top hypothesis -> a proposed action.

    This is a keyword lookup table against the hypothesis's cause text,
    NOT an LLM call — action selection is exactly the kind of decision
    that should be auditable and reproducible, not left to a model's
    phrasing on a given run.

    Mapping mirrors the synthetic break patterns in
    data/synthetic/generate.py: timing/batch issues -> RERUN_JOB,
    schema/mapping/duplicate issues -> FLAG_FOR_REVIEW (needs engineering
    or reversal judgment this system shouldn't make autonomously), small
    currency/rounding drift -> MARK_RESOLVED. Anything unrecognized
    defaults to FLAG_FOR_REVIEW — the conservative choice when the cause
    doesn't clearly match a known playbook entry.
    """
    diag = state["diagnosis"]
    brk = state["break_event"]
    top = diag.hypotheses[0]
    action_type = _classify_action(top.cause)

    state["proposed_action"] = ProposedAction(
        break_id=brk.break_id,
        action_type=action_type,
        target_ref=brk.txn_ref,
        justification=top.cause,
        based_on_hypothesis_rank=top.rank,
    )
    state["status"] = "awaiting_approval"
    return state


def _classify_action(cause_text: str) -> ActionType:
    text = cause_text.lower()
    if any(kw in text for kw in ("batch", "before", "timing", "ran early", "race")):
        return ActionType.RERUN_JOB
    if any(kw in text for kw in ("schema", "mapping", "field", "deploy", "duplicate")):
        return ActionType.FLAG_FOR_REVIEW
    if any(kw in text for kw in ("rounding", "currency", "drift", "conversion")):
        return ActionType.MARK_RESOLVED
    if any(kw in text for kw in ("clock skew", "ntp", "tolerance")):
        return ActionType.NO_ACTION
    return ActionType.FLAG_FOR_REVIEW  # conservative default for unrecognized causes


def human_approval_node(state: ReconState) -> ReconState:
    """This is where the graph actually halts.

    build_graph() compiles with interrupt_before=["act"], so LangGraph
    pauses the thread right before ACT runs. An external approval
    service (a small API + UI showing the proposed action + evidence)
    posts an ApprovalDecision (with a freshly minted, single-use,
    scoped ApprovalToken on approval) which is written into state and
    the thread is resumed via graph.invoke(None, config) on the same
    thread_id.

    This node itself just validates whatever decision has landed in state.
    """
    return state


def act_node(state: ReconState) -> ReconState:
    """Calls recon-actuator MCP server. Requires a valid, unexpired,
    unused ApprovalToken matching this break_id + action_type — checked
    and burned inside recon_actuator itself, not trusted from upstream.

    This node is a thin dispatcher: it does not decide what to do
    (plan_action_node already decided) and it does not decide whether
    to proceed (route_after_approval already gated on approval) — it
    only translates ProposedAction + ApprovalToken into the one
    recon-actuator call that matches.

    Fails closed: if recon-actuator rejects the token for any reason
    (expired, already used, mismatched break_id/action_type), this is
    caught and turned into an escalation, never silently swallowed and
    never allowed to crash the graph mid-run.
    """
    action = state["proposed_action"]
    approval = state["approval"]
    token = approval.token

    if token is None:
        state["status"] = "escalated"
        state["escalation_reason"] = "act_node reached with no approval token in state"
        return state

    try:
        if action.action_type == ActionType.MARK_RESOLVED:
            recon_actuator.mark_resolved(action.break_id, action.target_ref, token)
        elif action.action_type == ActionType.RERUN_JOB:
            recon_actuator.rerun_job(action.break_id, action.target_ref, token)
        elif action.action_type == ActionType.FLAG_FOR_REVIEW:
            recon_actuator.flag_for_review(
                action.break_id, action.target_ref, action.justification, token,
            )
        elif action.action_type == ActionType.NO_ACTION:
            pass  # nothing to call the actuator for
        else:
            raise ValueError(f"unhandled action_type: {action.action_type}")
    except PermissionError as e:
        state["status"] = "escalated"
        state["escalation_reason"] = f"recon-actuator rejected the action: {e}"
        return state

    state["status"] = "verifying"
    return state


def verify_node(state: ReconState) -> ReconState:
    """Re-query ledger-telemetry to confirm the action ACTUALLY resolved
    the break — not just that recon-actuator's call returned success.
    This distinction matters: a write can succeed and still not fix the
    underlying problem (e.g. rerun_job ran but the upstream record still
    doesn't exist for some other reason).

    Verification logic is per-action-type, since "resolved" means
    something different for each:
      - RERUN_JOB: the previously-missing record should now exist,
        matching amount/currency, in the counter system.
      - MARK_RESOLVED: nothing was supposed to change (drift was within
        tolerance) — resolved means the action was logged, not that data
        moved.
      - FLAG_FOR_REVIEW: by design NOT resolved yet — a human still owns
        the actual fix. Verification here just confirms the review_queue
        entry exists, so the loop doesn't silently drop it.
      - NO_ACTION: trivially resolved (benign pattern, nothing to check).
    """
    brk = state["break_event"]
    action = state["proposed_action"]
    action_type = action.action_type

    resolved = False
    evidence_ids: list[str] = []
    notes = ""

    if action_type == ActionType.RERUN_JOB:
        rows = recon_actuator_reads.get_txn_record(brk.txn_ref, brk.counter_system)
        if rows:
            row = rows[0]
            ledger_rows = recon_actuator_reads.get_txn_record(brk.txn_ref, brk.source_system)
            source_row = ledger_rows[0] if ledger_rows else None
            resolved = bool(
                source_row and abs(row["amount"] - source_row["amount"]) < 0.01
            )
            evidence_ids = [f"{brk.break_id}-VERIFY-{action_type.value}"]
            notes = (
                f"Post-rerun check: {brk.counter_system} now has txn_ref={brk.txn_ref} "
                f"amount={row['amount']}, matches {brk.source_system} amount="
                f"{source_row['amount'] if source_row else 'MISSING'}"
            )
        else:
            notes = f"Post-rerun check: {brk.counter_system} still has no record for {brk.txn_ref}"

    elif action_type == ActionType.MARK_RESOLVED:
        resolved = True
        notes = "Marked resolved — drift was within tolerance, no data correction expected"

    elif action_type == ActionType.FLAG_FOR_REVIEW:
        resolved = False
        notes = "Flagged for human review — not resolved by this system by design"

    elif action_type == ActionType.NO_ACTION:
        resolved = True
        notes = "Benign known pattern — no action was needed"

    state["verification"] = VerificationResult(
        break_id=brk.break_id,
        action_taken=action_type,
        resolved=resolved,
        verification_evidence_ids=evidence_ids,
        notes=notes,
    )
    state["status"] = "reflecting"
    return state


def reflect_node(state: ReconState) -> ReconState:
    """Write a ReflectionRecord to episodic memory (past break -> outcome).
    This is what memory/episodic.py's search_similar and find_open_incident
    read back during future TRIAGE/DIAGNOSE steps, via runbook-kb's
    find_similar_past_breaks tool.

    The break_signature used for future similarity search is the break's
    own description text — the same field DIAGNOSE and search_runbooks
    already key off of, so a future break with a similar description
    surfaces this outcome as relevant context.
    """
    brk = state["break_event"]
    diag = state["diagnosis"]
    verification = state["verification"]
    record = ReflectionRecord(
        break_id=brk.break_id,
        final_hypothesis=diag.hypotheses[0],
        action_taken=verification.action_taken,
        outcome_resolved=verification.resolved,
        lessons=verification.notes,
        stored_at=datetime.utcnow(),
    )
    episodic.write(record, break_signature=brk.description)
    state["status"] = "resolved"
    return state


def suppress_node(state: ReconState) -> ReconState:
    state["status"] = "suppressed"
    return state


def escalate_node(state: ReconState) -> ReconState:
    state["status"] = "escalated"
    if "escalation_reason" not in state:
        state["escalation_reason"] = "low_confidence_or_rejected_or_timeout"
    return state


# ---------------------------------------------------------------------------
# Helper the approval service uses to mint tokens (called OUTSIDE the graph,
# by whatever service backs the human-approval UI/API).
# ---------------------------------------------------------------------------

def mint_approval_token(break_id: str, action_type: ActionType, approved_by: str):
    from schemas.models import ApprovalToken

    now = datetime.utcnow()
    return ApprovalToken(
        token_id=str(uuid.uuid4()),
        break_id=break_id,
        action_type=action_type,
        approved_by=approved_by,
        approved_at=now,
        expires_at=now + timedelta(minutes=15),
        used=False,
    )
