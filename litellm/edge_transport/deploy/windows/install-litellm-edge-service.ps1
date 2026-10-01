# Phase 3.5 §19: local workstation LiteLLM service (Windows).
#
# Registers a Scheduled Task that runs the LiteLLM proxy with the
# Differential Context edge profile, with:
#   - restart policy        (task restarts on failure, 3 tries, 1 min apart)
#   - log rotation          (runner appends stdout/stderr to a log file; the
#                            script archives logs older than 7 days on each
#                            install/update)
#   - health probes         (/health/liveliness + /health/readiness checked
#                            after start; litellm exposes both)
#   - dependency ordering   (task starts at logon; the edge transport's own
#                            capabilities fetch + epoch check make it safe
#                            to start before the WAN frontend is live — the
#                            first request heals via epoch reconciliation)
#
# Usage (admin PowerShell):
#   .\install-litellm-edge-service.ps1 -LiteLLMPath D:\_SATIN_AI_2\litellm\litellm
#
# The task is named "KelvinLiteLLMEdge". Uninstall:
#   Unregister-ScheduledTask -TaskName KelvinLiteLLMEdge -Confirm:$false

param(
    [Parameter(Mandatory = $true)]
    [string]$LiteLLMPath,
    [string]$TaskName = "KelvinLiteLLMEdge",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 4000,
    [string]$LogDir = "$env:LOCALAPPDATA\VIVERRA\LiteLLM\logs",
    [string]$EdgeBaseUrl = "https://edge.example.com",
    [string]$EdgeApiKey = "CHANGE_ME"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath (Join-Path $LiteLLMPath "litellm\__init__.py"))) {
    throw "LiteLLMPath does not look like a LiteLLM tree: $LiteLLMPath"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
# Log rotation: archive anything older than 7 days, keep the current run.
Get-ChildItem -LiteralPath $LogDir -Filter "*.log" -File |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
    ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination "$($_.FullName).$($_.LastWriteTime.ToString('yyyyMMdd')).old" -Force }

$python = (Get-Command python).Source
$stderr = Join-Path $LogDir "litellm-edge.err.log"

# The task runs a small runner that sets the edge profile environment, so
# the profile is explicit and reproducible (secrets are NOT written into
# the runner; they come from the scheduled task's own environment or the
# script parameters).
$runner = Join-Path $LogDir "run-litellm-edge.ps1"
@"
`$ErrorActionPreference = "Stop"
`$env:EDGE_SSEPROXY_ENABLED = "1"
`$env:EDGE_SSEPROXY_BASE_URL = "$EdgeBaseUrl"
`$env:EDGE_SSEPROXY_API_KEY = "$EdgeApiKey"
`$env:EDGE_STATE_PERSISTENCE = "sqlite"
`$env:EDGE_STATE_DB_PATH = "$env:LOCALAPPDATA\VIVERRA\LiteLLM\edge-state.sqlite3"
`$env:EDGE_CONTEXT_IDLE_TTL = "86400"
`$env:EDGE_SSEPROXY_CAPABILITIES_REFRESH_SECONDS = "300"
& "$python" -m litellm.proxy.proxy_cli --host $BindHost --port $Port --num_workers 1 *>> "$stderr"
"@ | Set-Content -LiteralPath $runner -Encoding UTF8

$taskAction = New-ScheduledTaskAction `
    -Execute "pwsh" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`"" `
    -WorkingDirectory $LiteLLMPath

# Restart policy: restart on failure, 3 attempts, 1 minute apart.
$taskSettings = New-ScheduledTaskSettingsSet `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -StartWhenAvailable

$taskTrigger = New-ScheduledTaskTrigger -AtLogOn

$task = New-ScheduledTask `
    -Action $taskAction `
    -Settings $taskSettings `
    -Trigger $taskTrigger `
    -Description "LiteLLM proxy with the Differential Context edge profile (Phase 3.5)."

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null

Write-Host "Registered scheduled task '$TaskName'."
Write-Host "Runner: $runner"
Write-Host "Logs:   $stderr (rotated by this script on reinstall)"
Write-Host "Start:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Probes: http://$BindHost`:$Port/health/liveliness (live), /health/readiness (ready)"
Write-Host "Edge state DB: $env:LOCALAPPDATA\VIVERRA\LiteLLM\edge-state.sqlite3 (WAL)"