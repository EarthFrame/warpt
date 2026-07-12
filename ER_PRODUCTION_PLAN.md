# warpt ER — Production Readiness Plan

> **Audience:** Fable (executing agent) + the warpt team.
> **Goal:** Take the `mao-support` branch's ER daemon intelligence layer from "works on one laptop"
> to **borderline production-ready, fleet-scale** software that can run on GPU nodes in a data center,
> with an enterprise dashboard and a provider-abstracted, agentic diagnosis pipeline.
>
> **Source of truth for current behavior:** the code on `mao-support`. This plan describes the delta.

---

## Handoff — start here (kickoff for the executing agent)

You're picking up the warpt "ER daemon" production-readiness effort on branch `mao-support`.

**First, read this whole document** — it's the source of truth: current-state review (§1), target architecture (§2), the four locked decisions (§0), and the 6-phase roadmap (§4). Don't re-litigate the four decisions; if reality forces a change, flag it.

**Current state.** Phase 0 (foundation hardening) is complete: VitalsNurse subprocess supervision (auto-restart + health), a lock-tolerant `read_only_snapshot()` for status/inspect, sentinel-confidence suppression, `total_power_w` = CPU+GPU sum, and a flaky-test fix. Daemon suite green (94 passed); ruff/black clean. See the **§1.1 status table** for exactly what's done vs. deferred. Deferred on purpose: **D8** timezone→Phase 3, **D6** structured-output→Phase 2, repo-wide **mypy**. Confirm `git status` before starting.

**Your next task is Phase 1 — the LLM provider abstraction** (`warpt/daemon/llm/`, §4). Claude API is the default provider; a local pulled model (Ollama) and a self-hosted OpenAI-compatible cluster (70–300B) must be drop-in via config. **Land this before touching agent internals (Phase 2)** — §7 explains why (otherwise you rewrite the agents twice). Use the `claude-api` skill for current model IDs/params; keep API keys out of `config.yaml`. Propose your Phase 1 breakdown before implementing.

> **Status 2026-07-06: Phase 1 functionality landed** on `mao-support` — `warpt/daemon/llm/` (protocol, three providers, registry w/ legacy back-compat, budget wrapper, secrets), agents/pipeline/daemon rewired, D6 closed, wizard + packaging updated. See the §4 Phase 1 status block for the outstanding test work; `escalate_to` wiring stays in Phase 2 as planned. **Next: Phase 2.**

**Non-negotiable principles.** Node autonomy is sacred (never make node diagnosis depend on central). Prefer structured/tool output over parsing prose. Keep the degradation ladder intact (every new dependency needs a fallback rung). Ship the remediation seam **empty** — interface + policy + audit, **zero** state-changing actions.

**Repo conventions.** Run `pytest` via `/Library/Frameworks/Python.framework/Versions/3.12/bin/pytest`; gate with `ruff` + `black`. Integration tests use real in-memory DuckDB; mock `time.monotonic` for threshold tests; derive threshold timing from `DEFAULT_GPU_THRESHOLDS`. `conftest.py` sets logging to WARNING.

---

## 0. The four locked decisions

These framed every choice below. Do not silently re-litigate them; if reality forces a change, flag it.

| Axis | Decision | Implication for the plan |
|------|----------|--------------------------|
| **Topology** | Hybrid: node-autonomous **+** central aggregation feeding an enterprise dashboard | Nodes must keep working when central is down; central is additive, never a hard dependency for node-local diagnosis. |
| **Inference** | Provider abstraction. Claude API default; local pulled model & self-hosted 70–300B cluster are drop-in | An `LLMProvider` seam is a **Phase 1 blocker** — everything agentic depends on it. |
| **Autonomy** | Perfect read-only `observe → diagnose → report` now; **make room** for on-node actions later | Build the remediation *interface + policy + audit* now, ship **zero** state-changing actions. |
| **Integrations** | Deferred. CLI tool for now; no DCGM/k8s/Prometheus/Slurm yet | Abstract the telemetry source and packaging so these slot in later without a rewrite. |

---

## 1. Where the branch is today (grounded in the code)

**What exists and is genuinely good:**
- `VitalsNurse` — subprocess-polls `warpt monitor --no-tui --json`, ring-buffers snapshots, heartbeats, **sustained** threshold detection with reset-on-recovery.
- `ChargeNurse` — breach → `events` + `cases`, dedupes into open cases per GPU+metric, **async single-flight** dispatch via one worker thread + queue.
- `ChartNurse` — pure-SQL analytics (1h/24h/7d rolling avgs, hour-of-day mean/stddev, prior cases, 7d event count, deviation %) + LLM interpretation, with an `analyze_without_llm()` degradation path.
- `Attending` — Python-orchestrated diagnosis, config-driven triage order, JSON-parse-with-retry + fallback.
- `Scribe` — pure-Python report formatting, no LLM.
- `CaseFile` — DuckDB with a real forward-only migration system.
- `OllamaClient` — retry/backoff, `OllamaPermanentError` for non-retryable failures.
- `pipeline.py` — a clean **4-rung degradation ladder** (full → chart-only → raw analytics → phase-1 event).
- CLI: `daemon start/stop/status/er/inspect`; config at `~/.warpt/config.yaml`; `daemon` pip extra.
- ~2,700 lines of tests across every agent + the degradation ladder.

**This is a solid Phase-1/2 foundation.** The work below is enhancement and hardening, not a rewrite.

### 1.1 Defects & gaps to clear first (found in review)

Status legend: ✅ done in the Phase-0 pass · ⏭️ deliberately deferred (see note) · 🔜 scheduled in a later phase.

| # | Severity | Status | Location | Issue |
|---|----------|--------|----------|-------|
| D1 | 🔴 blocker | ✅ | `.gitignore` L157–168 | **Committed merge-conflict markers** — resolved. |
| D2 | 🟠 reliability | ✅ | `vitals_nurse.py` | Monitor subprocess death was silent (daemon "alive but blind"). Now a supervisor thread restarts it with exponential backoff, tracks `consecutive_failures`, exposes `is_healthy()`/`get_health()`, and escalates to a critical log after `max_consecutive_failures`. |
| D3 | 🟠 correctness | ✅ | `daemon_process.py` `get_status()` | Now reads through a shared `read_only_snapshot()` helper (copies the DB + WAL if the writer holds the lock) — `status` and `inspect` both use it; the duplicated copy logic in `inspect` was removed. |
| D4 | 🟡 UX/correctness | ✅ | Scribe + `inspect_cmd` | Sentinel/invalid confidence is suppressed at **both display sites** (shown only for a valid 0–100). The Attending still stores the sentinel until Phase-2 calibration. |
| D5 | 🟡 correctness | ✅ | `vitals_nurse.py` `_write_vitals` | `total_power_w` now sums CPU + all GPU power (None-safe), not just CPU power. |
| D6 | 🟡 robustness | 🔜 P2 | `attending.py` `_try_parse` | Fragile `json.loads`-of-prose + retry — replaced by structured/tool output in **Phase 2**. |
| D7 | 🟢 style | ✅ | `chart_nurse.py` | Inline `import json` hoisted to module top. |
| D8 | 🟡 correctness | ⏭️ P3 | `chart_nurse.py` / `monitoring.py` | **Deferred to Phase 3.** Verified: DuckDB's default session TZ is host-local and it interprets naive timestamps in that TZ, so the daemon is **internally consistent on a single node today** (`datetime.now()` local vs stored local `ts`). The real exposure is multi-node aggregation (nodes in different TZs writing local-naive timestamps). A correct fix is a coordinated UTC migration through the shared `monitoring.py` snapshot format + DuckDB `SET TimeZone='UTC'` + test updates — it belongs with the Phase-3 fleet store, not a hardening pass. |
| D9 | 🟢 hygiene | ⏭️ | branch name | `mao-support` doesn't describe the ER-daemon work; rename/retitle at PR time (owner's call). |

**Also fixed:** a pre-existing **flaky** hour-of-day test (`test_chart_nurse.py`) that failed in the first ~10 min of each clock hour (seed points spilled into the previous hour) — made deterministic with an in-hour anchor.

**Deferred baseline item:** repo-wide **mypy** gate (Phase-0 DoD) — holding off because the wider repo isn't type-clean yet and a full sweep is its own task; the new daemon code is annotated. Track separately.

**Status:** Phase 0 is **done** except the two deliberate deferrals (D8→P3, D6→P2) and the mypy gate. Full test suite green; ruff + black clean on all touched files.

**Deliverable of clearing these:** a green, conflict-free, self-restarting baseline to build on. ✅

---

## 2. Target architecture

```
┌────────────────────────────── GPU NODE (one per node, autonomous) ──────────────────────────────┐
│                                                                                                  │
│   Telemetry Source (abstract)          Intelligence Pipeline                    Reporting        │
│   ┌───────────────────────┐            ┌───────────────────────────────┐        ┌────────────┐   │
│   │ warpt monitor --json  │  vitals    │ ChartNurse  → Attending(agent) │  case  │  Scribe    │   │
│   │ (later: NVML/DCGM/AMD) │──────────▶ │  analytics    tool-use loop    │──────▶ │  report    │   │
│   └───────────────────────┘            │               evidence gather  │        └────────────┘   │
│         │  VitalsNurse (poll, HB,      │               structured out    │              │          │
│         │  sustained thresholds)       └───────────────────────────────┘              │          │
│         ▼                                        │ uses                                │          │
│   ChargeNurse (events+cases, async)      ┌───────▼─────────┐   ┌──────────────────┐    │          │
│         │                                │ LLMProvider     │   │ Remediation SEAM │    │          │
│         ▼                                │ (Claude default,│   │ propose→policy→  │    │          │
│   CaseFile (node-local DuckDB) ◀─────────│  local, cluster)│   │ audit  (NO exec) │    │          │
│         │                                └─────────────────┘   └──────────────────┘    │          │
│         └──────────── NodeReporter (buffered push; survives central outage) ───────────┘          │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                │ mTLS + auth (push vitals rollups, events, cases,
                                                │              agent-activity stream)
                                                ▼
                        ┌──────────────────── CENTRAL CONTROL PLANE ────────────────────┐
                        │  Ingest API (FastAPI)  →  Fleet Store (Postgres/Timescale)     │
                        │  Fleet-level diagnosis (optional heavy models)                 │
                        │  Dashboard API (REST + WebSocket/SSE live stream)              │
                        └───────────────────────────────┬───────────────────────────────┘
                                                         ▼
                        ┌──────────────────── ENTERPRISE DASHBOARD (web) ───────────────┐
                        │  Fleet heatmap · node/GPU health (green/amber/red) · live      │
                        │  agent-activity stream (what/why/insight) · case timelines +   │
                        │  reasoning chains · time-series · alerts · RBAC/SSO            │
                        └────────────────────────────────────────────────────────────────┘
```

**Key architectural rules:**
- **Node autonomy is sacred.** A node with no network still observes, diagnoses (via local model fallback), and stores cases. Central is a consumer, not a controller.
- **DuckDB stays node-local** (it's a single-writer analytical store — perfect for the node). The **fleet store is a real multi-writer TSDB** (Postgres + TimescaleDB, or equivalent). Don't try to make DuckDB the fleet database.
- **One seam per concern:** telemetry source, LLM provider, agent tools, and remediation actions are each behind an interface so future backends/models/actions plug in.

---

## 3. Cross-cutting production requirements (apply to every phase)

- **Config & secrets:** typed config (pydantic) with layered precedence (defaults → file → env). **API keys never in `config.yaml`** — read from env / secret file; redact in logs.
- **Observability of the agents themselves:** structured JSON logs with correlation IDs (per case, per pipeline run), and self-metrics (events/min, diagnosis latency, LLM tokens/cost, degradation-rung hit counts).
- **Reliability:** every long-lived loop is supervised and self-restarts; every external call (LLM, central, subprocess) has timeout + retry + circuit breaker.
- **Security:** node↔central is mTLS + token auth; dashboard is SSO/OIDC + RBAC; telemetry leaving the node to a cloud LLM is a **data-egress decision** — support redaction and an "offline/local-only" mode.
- **Auditability:** the full reasoning chain, every tool call, every piece of evidence, and every *proposed* action is persisted per case (required for trust and for the future remediation story).
- **Testability:** integration tests on real in-memory DuckDB (existing pattern), plus chaos tests (kill subprocess, kill central, LLM timeout/garbage), plus a fleet simulator (N synthetic nodes).
- **Cost control:** per-provider token budget + rate limit; tiered escalation so cheap/local handles the common case.

---

## 4. Phased roadmap

Each phase is independently valuable and ends in a demoable, tested state. Rough sequencing; Phases 3–5 can overlap once 1–2 land.

---

### Phase 0 — Foundation hardening *(fast; unblocks everything)*

**Objective:** green, conflict-free, self-healing baseline.

- Clear D1–D9 above. Specifically:
  - Resolve the `.gitignore` conflict.
  - **VitalsNurse subprocess supervision (D2):** detect subprocess death, restart with backoff, emit an event/alert on repeated failure. Add a watchdog so "daemon alive but blind" is impossible.
  - **Status without contention (D3):** route `get_status()` through the same read-only/copy path as `inspect`, or better, expose an in-process status channel from the running daemon (see Phase 6 health endpoint).
  - **Confidence hygiene (D4):** suppress sentinel confidence in Scribe until real calibration lands (Phase 2).
  - Fix D5/D7/D8; add a regression test per fix.
- Add `mypy` to the gate (ruff/black already present); type the daemon package.
- **DoD:** CI green (ruff, black, mypy, pytest); killing `warpt monitor` mid-run auto-recovers within N seconds and logs it; no sentinel values in any report; no conflict markers anywhere.

---

### Phase 1 — Inference provider abstraction *(core enabler — blocks Phase 2)*

> **Status 2026-07-06: ✅ functionality complete** (all modules below landed; agents/pipeline/daemon rewired; wizard + packaging updated; D6 closed via schema-validated diagnosis). **Outstanding testing before Phase 1 can be called done:**
> - `tests/test_llm_claude.py` — mocked SDK: exception mapping, structured output, refusal, missing-SDK/missing-key → `LLMPermanentError`
> - `tests/test_llm_openai_compat.py` — payload shape, bearer auth, `response_format` rejection → prompt-emulation fallback
> - `tests/test_llm_budget.py` — mocked `time.monotonic`: backoff sequence, breaker open/half-open/close, RPM window, token accounting
> - `tests/test_llm_secrets.py` — env/file resolution, inline-key refusal, redaction; `save_config` api_key strip
> - Cross-provider **contract suite** (same `generate()` semantics across all three backends — the "same diagnosis, three backends" DoD in test form)
> - Wizard flow test (`er_cmd`) with mocked prompts; one end-to-end degradation run with a dead provider
>
> Already covered: base types/validator, ollama provider, registry (incl. legacy back-compat), and the rewired agent/pipeline/daemon tests — all green.

**Objective:** swap Claude ⇄ local model ⇄ inference cluster via config, with no agent code changes.

**New module: `warpt/daemon/llm/`**
- `base.py` — `LLMProvider` protocol: `generate(messages, *, system, tools=None, response_schema=None, timeout, ...) -> LLMResponse`. Model-agnostic message shape; first-class support for **system prompt, tool-use, and structured/JSON-schema output** (Claude tool use / structured outputs; emulate for backends that lack it).
- `providers/claude.py` — Anthropic API. **Default provider.** Use tool-use / structured outputs for the Attending diagnosis (kills the fragile parse-retry, D6). Model IDs from the `claude-api` skill; default to a current model (e.g. Sonnet-class for cost, Opus-class for hard cases).
- `providers/ollama.py` — refactor the existing `OllamaClient` behind the protocol; keep `get_installed_models` / `ollama pull` UX for "pull a local model when needed."
- `providers/openai_compat.py` — OpenAI-compatible HTTP (vLLM / TGI / a self-hosted 70–300B cluster). This is the "plug in an inference cluster" story.
- `registry.py` — build providers from config; per-agent provider selection (Chart Nurse and Attending can differ).
- `budget.py` — token/cost accounting, per-provider rate limit, circuit breaker, structured retry (fold in the existing backoff).
- `secrets.py` — API keys from env/secret file; never persisted to `config.yaml`.

**Config shape (illustrative):**
```yaml
intelligence_enabled: true
llm:
  default: claude
  providers:
    claude:      { type: anthropic, model: claude-sonnet-5, api_key_env: ANTHROPIC_API_KEY }
    local:       { type: ollama, url: http://localhost:11434, model: llama3:8b }
    cluster:     { type: openai_compat, url: http://inference.internal/v1, model: llama3:70b }
  agents:
    chart_nurse: { provider: local }          # cheap/fast triage
    attending:   { provider: claude }         # high-quality diagnosis
    escalate_to: cluster                      # optional: hard cases → big model
```

- **Tiered escalation (design in now, wire in Phase 2):** small/local model triages; escalate low-confidence or high-severity cases to the big model. This is the natural home for the "70–300B when needed" capability.
- Update the `daemon er` wizard to configure providers (detect Claude key, detect Ollama, test cluster URL).

**DoD:** the same diagnosis runs unchanged against Claude, a local Ollama model, and an OpenAI-compatible endpoint by flipping config; structured output validated by schema (no free-form JSON parsing); keys never touch disk; token/cost metrics emitted.

---

### Phase 2 — Perfect the agentic `observe → diagnose → report` pipeline

> **Status 2026-07-07: ✅ implementation landed** on `er-experimentation` — 2.0 (tools seam: `ToolDef`/`ToolCall`, native Claude/Ollama/OpenAI-compat mappings + `tool_emulation.py` fallback), 2a (`agents/tools/` with 6 tools incl. policy+idle-gated probe), 2b (DIAGNOSIS_SCHEMA v2), 2c (`calibration.py`, `-1.23` sentinel retired → NULL on fallback), 2d (ChartNurse `correlated_signals`), 2e (versioned `prompts.py` + per-case `prompt_snapshot`), 2f (`remediation/` — policy deny-by-default, audit, zero executable actions), 2g (schema v2: `tool_calls`, `actions`, case columns; verified fresh + v1→v2 upgrade on DuckDB 1.5.1). Wired into `daemon_process`; Scribe renders v2 fields; probes **disabled by default**. Suite: 279 passed; **outstanding test updates** (deliberately deferred): `test_attending.py` (sentinel import), 2 × `test_casefile.py` migration-count asserts, plus new-module test coverage.

**Objective:** turn the Attending from a one-shot prompt into a real evidence-gathering **agentic loop**, and make the whole pipeline trustworthy, calibrated, and auditable.

> **Current-state note (grounded in code):** the Attending today is a **single `provider.generate(response_schema=…)` call** (`attending.py`), not a loop; confidence is the `-1.23` sentinel (`attending.py` `CONFIDENCE_SENTINEL`); Chart Nurse analyzes **only the breach metric**, one at a time (`chart_nurse.py`); `prompts.py` is flat constants. The Phase-1 `LLMProvider.generate()` seam has **no `tools` parameter** — `base.py` deferred it ("tools support deliberately absent until the Phase-2 agent loop"). So 2.0 below is a real prerequisite, not a given.

**2.0. Extend the LLM provider seam for tool-use *(do this FIRST — it gates 2a–2c)***
- Add `tools=[…]` to `LLMProvider.generate()` and a tool-call return shape to `LLMResponse` (e.g. `tool_calls`). This is the P1 "one seam, three backends" contract extended to tool-use — **land it before writing the Attending loop or you rewrite the agent twice** (same lesson as P1-before-P2).
- `providers/claude.py`: native Anthropic tool-use.
- `providers/ollama.py` + `providers/openai_compat.py`: **Python-orchestrated emulation** (prompt-driven JSON tool selection) so local/cluster models drive the same loop and the degradation ladder still holds.
- The `ResilientProvider` budget/breaker wrapper and `registry.provider_for_agent` selection carry over unchanged.

**2a. Agent tool interface (read-only now; the remediation seam reuses it)**
- **New module `warpt/daemon/agents/tools/`:** `base.py` (`Tool` protocol: name, JSON schema, `run(args) -> result`), `registry.py`, and read-only tools:
  - `query_historical_vitals` (windowed stats for a GPU/metric)
  - `get_current_snapshot` (latest ring-buffer frame)
  - `list_prior_cases` / `get_case`
  - `get_gpu_specs` (from `gpu_profiles`)
  - `run_diagnostic_probe` — wire the existing `warpt stress` runner as a **read-only** reproduction tool (respects the "idle-check before stress" rule from the design notes; never runs a stress test on a busy tenant GPU without policy clearance). This realizes the `stress_tests_ordered` / `stress_test_results` columns already in the schema.
- The Attending runs a bounded tool-use loop (Claude tool use, or Python-orchestrated for local models): gather evidence → hypothesize → optionally probe → conclude. Cap iterations + wall-clock + token budget.

**2b. Structured diagnosis (replaces D6)**
- Diagnosis is a schema-validated object: `hypothesis`, `confidence`, `severity`, `recommended_action` (structured, not prose — the future remediation input), `reasoning_chain`, `evidence[]`, `tools_used[]`. Validated at the provider layer; retry on schema-miss, not on `json.loads`.

**2c. Real confidence calibration (retires the `-1.23` sentinel)**
- Compute confidence from: LLM self-report **+** evidence quality (was a probe run? deviation magnitude vs baseline stddev? corroborating signals like throttle reasons + thermal + power?) **+** prior-case agreement. Document the formula; make it testable and deterministic given fixed inputs.

**2d. Multi-signal correlation**
- Chart Nurse should surface correlated signals together (throttle_reasons + temperature_c + power_w + utilization deviation) so the Attending reasons over a coherent picture, not one metric.

**2e. Auditability & prompt safety**
- Persist the full reasoning chain, evidence, and tool calls per case (schema already has `reasoning_chain`, `historical_context`; extend as needed).
- **Prompt-injection hygiene:** telemetry is trusted, but treat any string that could reach an action later as untrusted; version prompts (`prompts.py` → versioned, with a prompt registry) and snapshot the prompt+model per case for reproducibility.

**2f. Remediation seam — build the room, ship no actions**
- **New module `warpt/daemon/remediation/`:** `base.py` defines the `Action` lifecycle: `propose → policy_check → (approve) → execute → verify → rollback`. `policy.py` is an allow/deny/require-approval engine. `audit.py` writes every proposed action to an audit table.
- **In this phase, the pipeline only *proposes and records* actions** (structured `recommended_action` + policy verdict), never executes. When on-node actions are added later, they implement `execute()`/`rollback()` and flip policy — **no pipeline rewrite.**

**2g. Schema migration v2 (explicit — the current schema only has room, not tables)**
The v1 schema (`casefile.py` `_SCHEMA_V1`) has `cases.stress_tests_ordered/stress_test_results/reasoning_chain/historical_context` columns but **no** tables for tool calls or proposed actions. Add a forward-only migration:
- `tool_calls` (or a JSON column on `cases`) — per-case record of every tool invocation: tool name, args, result, timestamp — the auditability payload P3 ships and P4 renders.
- `actions` — the remediation audit table (2f): proposed action, policy verdict, status; **write-only in this phase, never executed.**
- Prompt/model-version columns (or snapshot JSON) on `cases` for replay (2e).
- Wire the read-only `run_diagnostic_probe` tool to actually populate `stress_tests_ordered`/`stress_test_results` (currently unused).

**Intra-phase ordering:** 2.0 (seam) → 2a (tools) → agentic loop → 2b (structured diagnosis v2) → 2c (calibration) → 2g (migration lands alongside 2a/2e/2f as those tables are needed). 2d and 2f can proceed in parallel once the seam exists.

**DoD:** Attending gathers evidence via tools and produces schema-valid, calibrated diagnoses; a diagnostic probe can run under policy (with idle-check) and its result feeds the case; every case has a replayable reasoning chain + prompt/model snapshot; proposed actions are recorded and policy-gated but never executed; degradation ladder still holds when the LLM/tools fail.

---

### Phase 3 — Fleet control plane *(node → central)*

> **Status 2026-07-07: ✅ implementation landed** on `er-experimentation` — `warpt/fleet/`: `node_reporter.py` (disk-buffered JSONL outbox w/ size cap, cursor state file, stable node identity, heartbeat; **verified**: central down → buffers, reconnect → backfills with zero duplicates), `messages.py` (schema-versioned pydantic), `central/` (FastAPI ingest + query API, SQLAlchemy fleet store — Postgres for prod / SQLite for dev, bearer-token auth w/ TLS/mTLS via uvicorn flags, SSE activity stream, `/healthz`+`/readyz`). Agent-activity stream derived from the Phase-2 `tool_calls`/`actions` audit tables. CLI: `warpt fleet serve|token`; new `fleet` pip extra. Wired into the daemon behind `fleet.enabled: false`. **Deferred:** timezone/UTC migration (D8 — still open), central-side heavy-model diagnosis, per-node tokens, test suite.

**Objective:** fleet-wide visibility without sacrificing node autonomy.

- **`warpt/fleet/node_reporter.py`** — a component in the daemon that pushes vitals rollups, events, cases, and the **agent-activity stream** to central. **Buffered + retried on disk;** when central is unreachable the node keeps working and backfills later. Node identity, registration, heartbeat.
- **`warpt/fleet/central/`** — ingest API (FastAPI) + fleet store. **Use Postgres + TimescaleDB** (or equivalent) for multi-node time-series; keep node-local DuckDB for local analytics. Node inventory, GPU inventory, fleet-level cases.
- **Transport & security:** gRPC or HTTP/2; **mTLS + token auth**; schema-versioned messages; backpressure handling.
- **Optional central diagnosis:** heavy models (the 70–300B cluster) can run fleet-level correlation/diagnosis centrally, complementing node-local triage (ties into Phase 1 tiering).
- **HA posture:** central is stateless app tier over a replicated DB; document the failure story (node autonomy covers central downtime).

**DoD:** N nodes report into central; killing central for 10 min loses **zero** node-local functionality and backfills on reconnect; all node↔central traffic is authenticated + encrypted; fleet store answers "show me every open critical case across the fleet."

---

### Phase 4 — Enterprise dashboard *(the big visible deliverable)*

> **Status 2026-07-07: ✅ v1 landed** — `warpt/fleet/central/dashboard/index.html`, served by central at `/`. Self-contained zero-build SPA (deliberate deviation from the React guidance: offline-datacenter constraint, no node toolchain in repo — swappable later): ward-board node grid w/ semantic health (validated colorblind-safe status palette, icon+label never color-alone), open-case rail → full case drawer (hypothesis, calibrated confidence, reasoning chain, evidence, tools used, policy-gated recommended action), per-node vitals small-multiples (single-axis, crosshair+tooltip), **live agent-activity stream** over SSE (fetch-streaming so the bearer token rides in headers), dark+light themes. **Deferred:** SSO/OIDC+RBAC, alert paging hooks, visual QA pass in a real browser.

**Objective:** an enterprise-grade, real-time window into the fleet **and into what the agents are thinking and why.**

**Backend:** dashboard API over the fleet store — REST for queries + **WebSocket/SSE** for the live agent-activity stream (watch a diagnosis happen in real time).

**Frontend (web app):**
- **Fleet overview:** node grid / GPU heatmap, health rollups, **semantic color** (green healthy / amber warning / red critical) with accessible, colorblind-safe palettes and dark+light themes (use the team's `dataviz` palette guidance).
- **Live agent-activity stream:** for each node, show what Vitals/Charge/Chart/Attending/Scribe are doing *right now*, **why** (the trigger), and the insight produced — the transparency the user explicitly asked for.
- **Case detail:** full timeline, reasoning chain, evidence, tools called, confidence, deviation vs baseline, recommended action (+ policy verdict). This is where the Phase 2 auditability pays off.
- **Time-series:** utilization/temp/power vs 1h/24h/7d baselines and hour-of-day profiles.
- **Alerts & notifications:** critical-case surfacing; hooks for future paging.
- **AuthN/Z:** SSO/OIDC + RBAC; per-team/tenant views.
- **Enterprise polish:** empty/loading/error states, keyboard nav, responsive, audit-friendly.

**Tech guidance:** FastAPI backend; React frontend with a real design system; live updates over SSE/WebSocket. Design quality matters here — invoke the `frontend-design` and `dataviz` skills when building UI, and calibrate scope with `artifact-design` for any mockups.

**DoD:** an operator opens the dashboard, sees the whole fleet's health at a glance, drills into a red node, and **watches the agent's reasoning stream live** with the evidence and recommended action — colors, states, and transitions all behave at enterprise quality.

---

### Phase 5 — Production hardening & operability

> **Status 2026-07-07: ✅ core slice landed** — `warpt/daemon/health.py` (in-process `/healthz`/`/readyz`/`/status`, supersedes the D3 snapshot hack on-node; behind `daemon_http.enabled: false`), `warpt/daemon/janitor.py` (vitals + closed-case retention w/ CHECKPOINT, config `retention.*`), `warpt daemon start --foreground` for supervised mode, `packaging/systemd/*.service` (auto-restart, resource caps, hardening), `packaging/docker/Dockerfile.central`, and `docs/er-operations.md` (deploy guides, config reference, runbooks: monitor death, LLM outage, DuckDB WAL recovery, central outage, token rotation, upgrades). **Deferred (deliberate):** chaos suite + fleet simulator (test scaffolding — excluded from this run), `/security-review` pass, release engineering (CI/CD, pinned deps, vuln scan), mypy gate.

**Objective:** things you only appreciate at 3am.

- **Process supervision & packaging:** systemd unit(s) with auto-restart + watchdog (reconcile with the existing power/rust-daemon integration noted in project memory); container image; documented resource limits so the daemon never starves tenant workloads.
- **Health/readiness endpoints:** liveness, readiness, and a rich status (replaces the D3 second-connection hack).
- **Data lifecycle:** vitals retention + rollup/compaction on the node; DB size caps + backpressure; fleet-store retention tiers.
- **Failure recovery:** DuckDB corruption/WAL recovery runbook; provider outage → local fallback verified; central outage → backfill verified.
- **Testing at scale:** chaos suite (kill subprocess, kill central, LLM timeout/garbage/schema-miss, disk full) + a **fleet simulator** running dozens of synthetic nodes for load/soak.
- **Security review:** run the `/security-review` skill over the new surfaces (central ingest, dashboard auth, secret handling, data egress).
- **Docs & runbooks:** operator guide, config reference, deployment guide, incident runbooks, and an architecture doc; upgrade/migration guide for the DuckDB schema and fleet store.
- **Release engineering:** versioning, pinned deps, dependency/vuln scanning, CI/CD, reproducible builds.

**DoD:** a documented, containerized, supervised deployment survives the chaos suite; a new operator can deploy a node + central + dashboard from the docs alone; security review has no open highs.

---

## 5. Production-readiness checklist (definition of "borderline production ready")

- [ ] **Reliability:** every loop supervised + self-restarting; no "alive but blind" state; node survives central + LLM outages.
- [ ] **Security:** mTLS node↔central; SSO/RBAC dashboard; secrets never on disk/in logs; offline/redaction mode for data egress.
- [ ] **Observability:** structured logs w/ correlation IDs; self-metrics; the dashboard.
- [ ] **Scalability:** N nodes into central; data retention/rollup bounded; fleet store is multi-writer.
- [ ] **Correctness:** structured, schema-validated diagnoses; calibrated confidence; no sentinels; timezone-pinned analytics.
- [ ] **Auditability:** replayable reasoning chain + prompt/model snapshot + proposed-action audit per case.
- [ ] **Extensibility:** telemetry-source, LLM-provider, agent-tool, and remediation-action seams all in place.
- [ ] **Testability:** integration + chaos + fleet-simulation suites green in CI.
- [ ] **Deployability:** container + systemd + docs + runbooks; clean upgrade path.
- [ ] **Cost:** per-provider token budget + rate limit + tiered escalation.

---

## 6. Deferred / open (revisit later)

- **Integrations:** DCGM/NVML native telemetry (drop subprocess parsing), Prometheus/OTel export, Kubernetes DaemonSet/operator + cordon/drain, Slurm/scheduler job context. Kept behind the telemetry-source + packaging seams so they slot in without a rewrite. Multi-vendor backends (AMD, Tenstorrent) already have branches — align the telemetry-source abstraction with them.
- **On-node remediation actions:** the *interface, policy engine, and audit log* land in Phase 2; the *actions themselves* (throttle, cordon, kill, restart, with approval + rollback) are a deliberate follow-on.
- **Central HA topology** (multi-region, DR) — size once fleet load is real.

---

## 7. Notes for Fable (execution guidance)

- **Respect node autonomy in every design choice.** If a feature makes the node depend on central to diagnose, it's wrong.
- **Land the LLM-provider seam (Phase 1) before touching agent internals (Phase 2)** — otherwise you'll rewrite the agents twice.
- **Ship the remediation seam empty.** The point is zero behavior change now and zero rewrite later. Resist implementing any state-changing action.
- **Prefer structured/tool output over prose parsing** everywhere an LLM feeds downstream logic.
- **Keep the degradation ladder intact** as you add capability — every new dependency needs a graceful-fallback rung.
- **Test with real in-memory DuckDB** (existing pattern), mock `time.monotonic` for thresholds, derive threshold-test timing from `DEFAULT_GPU_THRESHOLDS` (per the team's testing conventions).
- **Use the skills:** `claude-api` for model IDs/params, `frontend-design`+`dataviz`+`artifact-design` for the dashboard, `security-review` before shipping central/dashboard, `tdd` for the pipeline work.
- **This plan can be decomposed into GitHub issues** via the team's `prd-to-issues` flow — each phase is a milestone, each workstream a tracer-bullet slice.

---

_Living document — update as decisions land. Grounded in `mao-support` @ current HEAD._
