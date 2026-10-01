# LOCAL EDGE — LiteLLM Differential Context edge (Windows workstation)

Practical workflow for the local LiteLLM edge gateway that fronts SSEProxy's
Differential Context WAN endpoint.

## 1. Run Start-LiteLLM-Edge.cmd

Double-click `Start-LiteLLM-Edge.cmd` (repo root) or run:

```powershell
.\scripts\Start-LiteLLM-Edge.ps1
```

The launcher:

- reads `config\edge-production.local.yaml` (gitignored; copy the committed
  `config\edge-production.example.yaml` and fill in real values first);
- requires `EDGE_SSEPROXY_API_KEY` (from the environment or the local file);
- uses the dedicated venv at `.venv` (create it once with
  `uv venv --python 3.13 .venv`, then
  `uv pip install --python .venv\Scripts\python.exe "litellm[proxy]" zstandard`
  and
  `uv pip install --python .venv\Scripts\python.exe -e D:\_SATIN_AI_2\differential-context`).
  Note: an editable install of the fork source (`pip install -e .`) requires
  rustc >= 1.94 (litellm-rust maturin bridge); the wheel-based environment
  above installs the runtime dependencies and the local fork source is used
  automatically because the launcher runs from the repo root (`python -m`
  puts the current directory first on `sys.path`).
- binds `127.0.0.1:4000` with a single worker (the edge profile is
  process-local and MUST NOT run multi-worker);
- forces UTF-8 stdio for the child (`PYTHONUTF8=1`) — required, otherwise
  the LiteLLM startup banner crashes the proxy with `UnicodeEncodeError`
  (exit code 3) on a cp1252 Windows console;
- appends logs to `D:\_SATIN_AI_2\logs\litellm-edge\` (every line written
  immediately, never truncates);
- writes a PID file and refuses to start a duplicate instance;
- waits for `/health/liveliness` then `/health/readiness`;
- optionally probes the remote WAN `/v1/transport/capabilities` (tolerates
  the self-signed bench certificate) and reports unavailability clearly;
- then STAYS ATTACHED as the supervisor: keep the window open, live logs
  stream to the console and the log files; stop with Ctrl+C in that window
  or with `Stop-LiteLLM-Edge.ps1` from another window.

Use `.\scripts\Start-LiteLLM-Edge.ps1 -DryRun` to validate everything without
launching anything.

## 2. Point coding agents to http://127.0.0.1:4000/v1

- Base URL for agents: `http://127.0.0.1:4000/v1`
- Route model: `differential_sseproxy` (requests whose `model` equals this, or
  starts with `differential_sseproxy/`, take the edge transport).
- The edge transport uses the shared `differential-context` package for all
  protocol logic; LiteLLM only owns transport concerns.
- The WAN boundary is the Caddy TLS + ALPN h2 frontend in front of SSEProxy.
  Strict HTTP/2 is enforced on the edge route: a negotiated HTTP/1.1 raises a
  transport error — there is no silent downgrade.

## 3. Verify Status-LiteLLM-Edge.ps1

```powershell
.\scripts\Status-LiteLLM-Edge.ps1
```

Shows: process state, local liveness/readiness, local port, edge SQLite DB
path, remote capability status (best effort), and negotiated transport info
from the local `/metrics` surface (best effort).

## 4. Stop with Stop-LiteLLM-Edge.ps1

```powershell
.\scripts\Stop-LiteLLM-Edge.ps1
```

Reads the PID file, verifies the process really is this LiteLLM edge launch,
requests graceful termination, waits a bounded window, forces only if needed,
then removes the PID file.

---

## Logs

- `D:\_SATIN_AI_2\logs\litellm-edge\litellm-edge.out.log` — stdout (appended)
- `D:\_SATIN_AI_2\logs\litellm-edge\litellm-edge.err.log` — stderr (appended)
- `D:\_SATIN_AI_2\logs\litellm-edge\litellm-edge.pid` — PID file (pid, start
  time, command line, port)

## SQLite edge-state location

- Default: `%LOCALAPPDATA%\VIVERRA\LiteLLM\edge-state.sqlite3` (WAL mode,
  durable acknowledged edge state + pending in-flight transitions). Override
  with `EDGE_STATE_DB_PATH` in the local config.

## Local secret/config location

- `config\edge-production.local.yaml` — gitignored local runtime config.
  All server URL/token/secret values come from this file or from environment
  variables (`EDGE_SSEPROXY_BASE_URL`, `EDGE_SSEPROXY_API_KEY`).
- Never put real keys into `edge-production.example.yaml` (committed).

## Rollback behavior

- Stop the edge (`Stop-LiteLLM-Edge.ps1`) — agents pointed at
  `127.0.0.1:4000` immediately lose the edge route; the WAN chain (SSEProxy +
  engine) is untouched and keeps serving direct traffic.
- Nothing in this repo changes remote routing. SSEProxy/Caddy/engine on the
  benchmark node are unaffected by anything on this workstation.
- The pre-production timestamped backup of every tree (SSEProxy, LiteLLM,
  differential-context, benchmark artifacts, local runtime configs) is at
  `D:\_SATIN_AI_2\_backups\20261001-205802\` with `MANIFEST\MANIFEST.md`.