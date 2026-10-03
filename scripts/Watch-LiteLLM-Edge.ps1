# =============================================================================
# Watch-LiteLLM-Edge.ps1 — continuous watchdog for the local Differential
# Context edge (127.0.0.1:4000).
#
# Loop (infinite):
#   1. probe TCP listener + /health/liveliness every $IntervalSeconds
#   2. on failure: kill any lingering litellm.proxy.proxy_cli processes,
#      remove stale PID file, relaunch via the DETACHED launcher
#   3. log every transition to the watchdog log
#
# The watchdog itself is meant to run as a persistent background process
# (survives console/session close). Stop it by killing this PID or via
# the background process manager.
#
# SGLang and remote routing are never touched.
# =============================================================================

[CmdletBinding()]
param(
    [int]$IntervalSeconds = 15
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$Port = 4000
$LogFile = "D:\_SATIN_AI_2\logs\litellm-edge\watchdog.log"
$DetachedLauncher = "D:\_SATIN_AI_2\LiteLLM\scripts\Start-LiteLLM-Edge-Detached.ps1"

function Write-WatchdogLog {
    param([string]$Msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    try { Add-Content -LiteralPath $LogFile -Value $line } catch { }
}

function Test-EdgeAlive {
    $listener = Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $listener) { return $false }
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health/liveliness" -UseBasicParsing -TimeoutSec 5
        return ($resp.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Clear-EdgeProcesses {
    # Force-kill every local process whose command line identifies it as the
    # LiteLLM edge (launcher + worker). Safe: the edge is local-only.
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'litellm\.proxy\.proxy_cli' }
    foreach ($p in $procs) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            Write-WatchdogLog ("killed leftover edge process PID " + $p.ProcessId)
        } catch {
            Write-WatchdogLog ("failed to kill PID " + $p.ProcessId + ": " + $_.Exception.Message)
        }
    }
    Start-Sleep -Seconds 2
}

$PidFile = "D:\_SATIN_AI_2\logs\litellm-edge\litellm-edge.pid"
if (Test-Path -LiteralPath $PidFile) {
    try { Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue } catch { }
}

Write-WatchdogLog "watchdog started (interval ${IntervalSeconds}s, port ${Port})"

$consecutiveFailures = 0
while ($true) {
    try {
        if (Test-EdgeAlive) {
            if ($consecutiveFailures -gt 0) {
                Write-WatchdogLog "edge recovered (was down for $consecutiveFailures check(s))"
            }
            $consecutiveFailures = 0
        } else {
            $consecutiveFailures++
            Write-WatchdogLog "EDGE DOWN (check #$consecutiveFailures) - cleaning and restarting"
            Clear-EdgeProcesses
            try {
                & $DetachedLauncher | Out-Null
                Write-WatchdogLog "detached launcher invoked (exit code $LASTEXITCODE)"
            } catch {
                Write-WatchdogLog ("launcher failed: " + $_.Exception.Message)
            }
            Start-Sleep -Seconds 5
            if (Test-EdgeAlive) {
                Write-WatchdogLog "EDGE RESTARTED OK"
                $consecutiveFailures = 0
            } else {
                Write-WatchdogLog "edge still down after restart - will retry"
            }
        }
    } catch {
        Write-WatchdogLog ("watchdog loop error: " + $_.Exception.Message)
    }
    Start-Sleep -Seconds $IntervalSeconds
}