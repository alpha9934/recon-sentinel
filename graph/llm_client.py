"""
The ONE LLM call in Recon Sentinel.

diagnose_with_llm() is the only function anywhere in this codebase that
calls a language model. Everything upstream (TRIAGE) and downstream
(PLAN ACTION, ACT, VERIFY) is deterministic Python / MCP tool calls.

Structured output is enforced by forcing a single tool call whose input
schema matches DiagnosisResult exactly (Pydantic -> JSON schema), so the
model literally cannot return free text here — the API rejects anything
that doesn't validate against the schema before it ever reaches us.
"""
from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from schemas.models import DiagnosisResult, EvidenceBundle

# Loads variables from a .env file in the repo root into os.environ, if
# present. Safe to call even if .env doesn't exist or a var is already
# set via real environment variables (those take precedence either way
# unless override=True, which we deliberately don't pass).
load_dotenv()

MODEL = "claude-sonnet-4-6"

# Below this overall_confidence, the model is instructed to (and
# route_after_diagnose enforces) escalate rather than let a shaky
# hypothesis flow into PLAN ACTION / an approval request. Tuned initially
# against the golden dataset in eval/harness.py — treat this as a knob,
# not a constant carved in stone.
CONFIDENCE_ESCALATION_THRESHOLD = 0.55

SYSTEM_PROMPT = """You are the DIAGNOSE step in Recon Sentinel, an agentic \
system that triages financial reconciliation breaks (mismatches between a \
core ledger, a payment gateway, and a settlement system).

You will be given a break description and a numbered list of evidence \
items, each with a stable evidence_id. Your job is ONLY to correlate that \
evidence into 1-3 ranked root-cause hypotheses. You do not take any \
action and you do not decide what action should be taken — that happens \
in a separate, deterministic step after you respond.

Rules you must follow:
1. Every hypothesis MUST cite at least one evidence_id from the list you \
were given, verbatim. Never invent an evidence_id. Never cite an \
evidence_id that doesn't actually support the hypothesis.
2. Rank hypotheses 1 (most likely) to N, sequential, no gaps.
3. confidence is your calibrated belief this hypothesis is correct, 0.0-1.0. \
Do not default to a "safe" middle value — if the evidence is genuinely \
weak or contradictory, say so with a low confidence number, don't hedge \
by writing an uncertain-sounding cause at high confidence.
4. Set requires_escalation=true if overall_confidence would fall below \
0.55, if the evidence is too thin or contradictory to support any \
hypothesis, or if you are not confident enough to recommend to a human \
reviewer that a specific action be taken.
5. reasoning must show the actual evidence-to-conclusion chain, not just \
restate the cause. A reviewer with no other context should be able to \
follow why the cited evidence supports the conclusion.
6. Never propose or reference an action (rerun a job, mark resolved, \
etc.) — that is out of scope for this step.
"""


def _format_evidence(evidence: EvidenceBundle) -> str:
    lines = [f"Break ID: {evidence.break_id}", "", "Evidence:"]
    if not evidence.items:
        lines.append("(no evidence items were gathered)")
    for item in evidence.items:
        ts = item.timestamp.isoformat() if item.timestamp else "no timestamp"
        lines.append(
            f"- evidence_id={item.evidence_id} | source={item.source_server} "
            f"| kind={item.kind} | ts={ts}\n  content: {item.content}"
        )
    return "\n".join(lines)


def _diagnosis_tool_schema() -> dict:
    """DiagnosisResult's Pydantic JSON schema, adapted into an Anthropic
    tool definition. Forcing tool_choice on this tool is what makes
    structured output non-optional rather than merely requested."""
    schema = DiagnosisResult.model_json_schema()
    # Anthropic tool schemas don't want a top-level "title"/"$defs" quirk
    # to trip up strict validators on some client versions; keep as-is,
    # Pydantic v2's schema is valid JSON Schema and works directly.
    return {
        "name": "submit_diagnosis",
        "description": "Submit the root-cause diagnosis for this reconciliation break.",
        "input_schema": schema,
    }


class DiagnoseLLMError(Exception):
    """Raised on any failure to obtain a valid DiagnosisResult — callers
    (diagnose_node) MUST treat this as 'escalate', never as 'proceed with
    nothing'. Fail closed, not open."""


def diagnose_with_llm(evidence: EvidenceBundle, break_description: str) -> DiagnosisResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise DiagnoseLLMError(
            "ANTHROPIC_API_KEY not set — cannot call the diagnosis model. "
            "Set it in your environment (e.g. via .env + python-dotenv) before "
            "running the graph past TRIAGE."
        )

    client = anthropic.Anthropic(api_key=api_key)
    tool = _diagnosis_tool_schema()

    user_content = (
        f"Break description: {break_description}\n\n"
        f"{_format_evidence(evidence)}"
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit_diagnosis"},
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.APIError as e:
        raise DiagnoseLLMError(f"Anthropic API call failed: {e}") from e

    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    if not tool_use_blocks:
        raise DiagnoseLLMError("Model did not return a tool_use block")

    raw_input = tool_use_blocks[0].input
    try:
        result = DiagnosisResult.model_validate(raw_input)
    except ValidationError as e:
        raise DiagnoseLLMError(f"Model output failed schema validation: {e}") from e

    # Belt-and-suspenders: even though the prompt instructs the model to
    # only cite real evidence_ids, verify it here rather than trusting it.
    # An LLM confidently citing a nonexistent evidence_id is exactly the
    # "silent-and-plausible" failure mode this system is designed to catch.
    valid_ids = {item.evidence_id for item in evidence.items}
    for hyp in result.hypotheses:
        bad_ids = set(hyp.cited_evidence_ids) - valid_ids
        if bad_ids:
            raise DiagnoseLLMError(
                f"Hypothesis rank {hyp.rank} cited nonexistent evidence_id(s): "
                f"{bad_ids} — treating as an ungrounded diagnosis, escalating."
            )

    # Enforce the escalation threshold ourselves too — don't just trust the
    # model set requires_escalation correctly.
    if result.overall_confidence < CONFIDENCE_ESCALATION_THRESHOLD:
        result = result.model_copy(update={"requires_escalation": True})

    return result
