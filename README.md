# Recon Sentinel

**An agentic system that triages financial reconciliation breaks** —
mismatches between a core banking ledger, a payment gateway, and a
downstream settlement system — the kind of operational problem every
multi-system BFSI institution hits nightly.

Personal project, independently designed and built on synthetic data.
Not a reproduction of any employer's system, but recognizably the same
class of problem you'd find in BFSI messaging/settlement operations.

---

## 🎥 Demo video

<!--
  RECORD YOUR DEMO, THEN REPLACE THIS SECTION.

  Easiest path (renders natively in GitHub's README preview):
    1. Open this README.md file directly in the GitHub web UI editor
       (or drag-and-drop into a new GitHub Issue/PR comment box — either
       surface accepts file uploads).
    2. Drag your recorded demo.mp4 (or demo.gif) into the edit box.
       GitHub uploads it and auto-inserts a working link like:
         https://github.com/user-attachments/assets/xxxxxxxx-xxxx-xxxx
    3. Copy that generated line into this README in place of the
       placeholder below, then commit.

  Local/offline alternative: save the recording to docs/demo.mp4 in this
  repo and reference it with a plain relative link — it won't play
  inline everywhere (e.g. it won't autoplay on PyPI or some renderers),
  but it's clickable and downloadable on GitHub:

    [Watch the demo](docs/demo.mp4)

  Recording checklist (matches demo.py's narrated sections):
    - [ ] MONITOR -> TRIAGE: real evidence gathered from the 3 read MCP servers
    - [ ] DIAGNOSE: the one real LLM call, ranked hypotheses with cited evidence
    - [ ] PLAN ACTION: deterministic action classification
    - [ ] HUMAN APPROVAL: the halt — narrate that this is a real graph
          interrupt with checkpointed state, not a prompt-level instruction
    - [ ] Approve it (type `y`), show the resume
    - [ ] ACT -> VERIFY: real DB write, real re-check that it actually worked
    - [ ] REFLECT: persisted to episodic memory
    - [ ] (bonus) run demo.py --reject once to show the escalation path too

  Suggested recording command:
    python3 demo.py --break-id BRK-TM-012
-->


[▶️ Watch the demo](https://github.com/alpha9934/recon-sentinel/blob/master/docs/demo.mp4)

---

## What it does

1. **Detects** a reconciliation break
2. **Gathers evidence** from multiple sources (transaction records, batch
   job status, deploy/schema-change events, logs, runbooks, past incidents)
3. **Proposes a ranked, evidence-cited root-cause hypothesis** with a
   calibrated confidence score
4. **Proposes a resolution action** — but **never executes it without
   human approval**
5. **Verifies** the fix actually worked (not just that the write succeeded)
6. **Learns** from the outcome for future similar breaks

## Architecture

A **bounded LangGraph state machine** — not a free-form ReAct agent loop.

```
MONITOR → TRIAGE → DIAGNOSE → PLAN ACTION → HUMAN APPROVAL (interrupt)
   → ACT → VERIFY → REFLECT
     ↳ SUPPRESS (dupe/benign)      ↳ ESCALATE (reject/timeout/low-confidence)
```

**Three design decisions carry the whole architecture:**

1. **State graph, not ReAct.** Deterministic transitions, bounded steps,
   a clean audit boundary — you can always answer "what states can this
   system be in," which you can't with a free-form loop.

2. **The LLM does one job only: DIAGNOSE.** Everything else (dedupe,
   evidence gathering, action classification, rollback, audit writes) is
   deterministic code. The LLM correlates evidence into a ranked
   hypothesis with cited evidence IDs — nothing more. See
   `graph/llm_client.py` — the only file in the codebase that calls an LLM.

3. **Human approval is a real graph interrupt**, not a prompt asking the
   model to "be careful." The graph halts, checkpoints state via
   LangGraph's `interrupt_before`, and only resumes on an external
   approval event that mints a single-use, scoped token — validated and
   burned inside the write-gated actuator itself, not trusted from
   upstream. See `graph/build.py` and `mcp_servers/recon_actuator/server.py`.

**MCP tool layer, split by trust boundary:**

| Server | Exposes | Trust |
|---|---|---|
| `ledger-telemetry` | txn records, batch status, break feed | read-only |
| `obs-context` | logs, traces, deploy/schema-change events | read-only |
| `runbook-kb` | retrieval over runbooks + past incidents (episodic memory) | read-only |
| `recon-actuator` | mark-resolved, re-run job, flag-for-review | write — approval-gated |

The write server is a physically separate module with the only writable
database connection in the whole codebase — the diagnosis path is
structurally incapable of touching a ledger record. That's an
architectural fact, not a policy the model is trusted to follow.

## Tech stack

| Layer | Tech |
|---|---|
| Orchestration | **LangGraph** (state machine, checkpointer, interrupts) |
| Tool integration | **MCP** (Model Context Protocol) — 4 servers, trust-split |
| LLM | **Anthropic API**, structured-output via forced tool-use (Pydantic-enforced) |
| Observability | **LangFuse** (per-node trace: prompt, evidence, confidence, cost, latency) |
| Evaluation | Custom harness — RCA top-1/top-3 accuracy, confidence calibration, false-action rate, faithfulness proxy |
| Data | Synthetic golden incident dataset (evidence bundle → known true cause) |
| Memory | 3-tier — working state (per-incident), episodic (past break→outcome), semantic (runbook corpus) — kept strictly separate |
| Language | Python 3.12+ |

## Repo structure

```
recon-sentinel/
├── graph/
│   ├── state.py              # ReconState — the single typed state object
│   ├── nodes.py               # node bodies (only diagnose_node calls an LLM)
│   ├── build.py               # StateGraph wiring, conditional edges, interrupt
│   ├── evidence_gathering.py  # TRIAGE's real evidence collection
│   ├── llm_client.py          # the ONE LLM call, structured output + safety guards
│   └── tracing.py             # optional LangFuse tracing wrapper
├── schemas/
│   └── models.py              # Pydantic contracts for every stage's I/O
├── mcp_servers/
│   ├── ledger_telemetry/      # read-only
│   ├── obs_context/           # read-only
│   ├── runbook_kb/            # read-only (runbooks + episodic memory retrieval)
│   └── recon_actuator/        # write, approval-gated — the only writable DB connection
├── memory/
│   └── episodic.py            # past break -> outcome, separate from semantic tier
├── eval/
│   ├── harness.py             # runs the real pipeline against golden data, scores it
│   ├── faithfulness.py        # lightweight word-overlap faithfulness proxy
│   └── generate_report.py     # turns a report JSON into a one-page markdown summary
├── data/synthetic/
│   ├── generate.py            # synthetic data generator (SQLite + golden dataset)
│   ├── recon.db               # generated — ledger/gateway/settlement/logs/etc.
│   ├── golden_incidents.jsonl # generated — 60 seeded breaks with ground truth
│   └── runbooks.json          # seed runbook corpus
├── demo.py                    # narrated live demo — real evidence, real LLM call,
│                               # real approval halt, real DB writes
├── run_local.py                # minimal smoke-test runner for the graph
├── test_stage*.py              # per-stage test suites (86+ checks total)
├── .github/workflows/eval.yml # CI regression gate
└── BUILD_PLAN.md               # the 8-stage build plan this repo followed
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate the synthetic dataset (SQLite DB + golden incidents)
python3 -m data.synthetic.generate

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

## Running it

**Run the test suites** (86+ checks across all 7 build stages, no API key
required except where noted):

```bash
python3 test_stage1_stage2.py   # synthetic data + read-only MCP servers
python3 test_stage3.py          # TRIAGE evidence gathering
python3 test_stage4.py          # DIAGNOSE fail-closed behavior + mocked safety guards
python3 test_stage4.py --live   # + a real call against the live model
python3 test_stage5.py          # PLAN ACTION, recon-actuator, approval interrupt
python3 test_stage6.py          # VERIFY, REFLECT, episodic memory
python3 test_stage7.py          # eval harness scoring logic
```

**Run the live demo:**

```bash
python3 demo.py                        # walks through a real seeded break
python3 demo.py --break-id BRK-SB-001  # a schema-break case (FLAG_FOR_REVIEW path)
python3 demo.py --reject               # demo the rejection/escalation path
python3 demo.py --reset                # rehearse the same break again from a clean state
```

**Run the eval harness against the golden dataset:**

```bash
python3 eval/harness.py --limit 10     # quick sample (still calls the real API)
python3 eval/harness.py                # full 60-incident run
python3 eval/generate_report.py        # turns the report into a one-pager
```

## Evaluation results

A 10-incident sample run (see `eval/EVAL_SUMMARY.md` for the full
generated report once you run the harness yourself):

| Metric | Result | Threshold |
|---|---|---|
| RCA top-1 accuracy | 70.0% | ≥60% ✅ |
| RCA top-3 accuracy | 80.0% | ≥85% (small-sample shortfall — see note) |
| False-action rate | 0.0% | ≤25% ✅ |
| Faithfulness (proxy) | 41.2% | ≥50% (proxy limitation — see note) |

**Two honest notes, not caveats to hide:**
- At n=10, a single incident swings top-3 accuracy by 10 points — this
  narrowly missed on small-sample noise, not a real regression. The
  threshold is tuned for the full 60-incident run.
- The faithfulness score is a **lightweight word-overlap proxy**
  (`eval/faithfulness.py`), not full RAGAS/DeepEval LLM-judged scoring.
  It catches pure fabrication well but penalizes legitimate paraphrasing
  — a correct, well-explained hypothesis that doesn't repeat the
  evidence's exact wording can score low here despite being genuinely
  grounded. It's a free per-commit CI signal, not a substitute for a
  periodic real RAGAS pass.

The CI gate correctly failed on this run for exactly these two reasons —
proof it's a real gate, not a cosmetic one.

## Known limitations / what's intentionally out of scope

- MCP servers run as in-process function calls, not separate OS processes
  over stdio/SSE transport — the trust-boundary *code* separation is real
  (only `recon_actuator` opens a writable DB connection), but the
  *process* separation described in the architecture doc is a
  straightforward mechanical extension, not yet done.
- `find_similar_past_breaks` / `search_runbooks` use keyword-overlap
  (Jaccard) retrieval, not real embeddings — documented upgrade path in
  both modules' docstrings.
- No real production deployment — evaluated offline against a synthetic
  golden dataset, built to demonstrate architecture ownership, not as a
  production claim.

## How to talk about this in an interview

See `Recon-Sentinel-Quick-Reference.md` for the full opening framing,
"is this related to your work?" answer, and the three architecture
decisions worth walking through unprompted. Short version: lead with the
architecture decisions (bounded state graph, LLM-does-one-job, structural
read/write separation), not the tool list — and treat a CI gate failure
like the one above as evidence the evaluation harness works, not
something to hide.
