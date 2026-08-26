"""
Recon Sentinel state graph.

    MONITOR -> TRIAGE -> DIAGNOSE -> PLAN_ACTION -> HUMAN_APPROVAL (interrupt)
       -> ACT -> VERIFY -> REFLECT
         |-> SUPPRESS   (dupe / benign known pattern, from TRIAGE)
         |-> ESCALATE   (reject / timeout / low confidence)

Deterministic-vs-LLM split is enforced by construction: only diagnose_node
calls an LLM. Every other node is plain Python / MCP tool calls.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.nodes import (
    act_node,
    diagnose_node,
    escalate_node,
    human_approval_node,
    monitor_node,
    plan_action_node,
    reflect_node,
    suppress_node,
    triage_node,
    verify_node,
)
from graph.state import ReconState


def route_after_triage(state: ReconState) -> str:
    ev = state["evidence"]
    if ev.is_duplicate or ev.is_benign_known_pattern:
        return "suppress"
    return "diagnose"


def route_after_diagnose(state: ReconState) -> str:
    if state["diagnosis"].requires_escalation:
        return "escalate"
    return "plan_action"


def route_after_approval(state: ReconState) -> str:
    decision = state["approval"].decision
    if decision == "approved":
        return "act"
    # rejected or timeout both escalate to a human queue
    return "escalate"


def build_graph():
    g = StateGraph(ReconState)

    g.add_node("monitor", monitor_node)
    g.add_node("triage", triage_node)
    g.add_node("diagnose", diagnose_node)
    g.add_node("plan_action", plan_action_node)
    g.add_node("human_approval", human_approval_node)
    g.add_node("act", act_node)
    g.add_node("verify", verify_node)
    g.add_node("reflect", reflect_node)
    g.add_node("suppress", suppress_node)
    g.add_node("escalate", escalate_node)

    g.set_entry_point("monitor")

    g.add_edge("monitor", "triage")
    g.add_conditional_edges("triage", route_after_triage,
                             {"suppress": "suppress", "diagnose": "diagnose"})
    g.add_conditional_edges("diagnose", route_after_diagnose,
                             {"escalate": "escalate", "plan_action": "plan_action"})
    g.add_edge("plan_action", "human_approval")
    # human_approval is where execution halts (see nodes.py: interrupt()).
    g.add_conditional_edges("human_approval", route_after_approval,
                             {"act": "act", "escalate": "escalate"})
    g.add_edge("act", "verify")
    g.add_edge("verify", "reflect")

    g.add_edge("reflect", END)
    g.add_edge("suppress", END)
    g.add_edge("escalate", END)

    # Checkpointer is what makes the human-approval interrupt real: state is
    # persisted, execution halts, and only an external event (approval
    # service posting a decision + token) resumes the thread.
    # Swap MemorySaver for a Postgres/Redis checkpointer in production.
    #
    # The interrupt is BEFORE human_approval, not before act: human_approval
    # is a trivial pass-through node (it just validates whatever decision
    # has already landed in state), and route_after_approval immediately
    # follows it. If the interrupt were placed before act instead,
    # human_approval would run to completion on the very first pass —
    # before any external approval event has had a chance to write
    # state["approval"] — and route_after_approval would find nothing
    # there to route on. Halting right before human_approval means the
    # graph pauses exactly at the point a decision is actually required,
    # and resumes into human_approval once it exists.
    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])


graph = build_graph()
