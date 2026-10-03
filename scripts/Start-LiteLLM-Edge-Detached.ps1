# =============================================================================
# Start-LiteLLM-Edge-Detached.ps1 — start the local Differential Context edge
# DETACHED from any console window.
#
# Problem this solves: the regular Start-LiteLLM-Edge.ps1 runs the edge as a
# child of the launching terminal; when that terminal window is closed,
# Windows sends CTRL_CLOSE_EVENT to the whole console process group and the
# edge dies SILENTLY (no traceback, no log line). Observed repeatedly on
# 2026-10-02 (edge instances 39432, 48700, 53216 all died when their hosting
# PowerShell window was destroyed).
#
# This launcher starts the ORIGINAL Start-LiteLLM-Edge.ps1 in its own hidden
# console via Start-Process -WindowStyle Hidden (UseShellExecute, no redirect
# here — the inner script owns the log pump). The hidden console is not
# attached to the caller's console, so closing the caller's terminal cannot
# reach it. The inner script keeps its full behavior: env resolution, pyc
# cleanup, PID file, liveness/readiness probes, log pump, WaitForExit.
#
# Usage:
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\Start-LiteLLM-Edge-Detached.ps1
#   (optionally -DryRun to only print what would be launched)
#
# Stopping is unchanged: scripts\Stop-LiteLLM-Edge.ps1 (reads the same PID
# file, which the inner script writes).
#
# This script NEVER touches SGLang, the remote SSEProxy chain, or any
# production routing. It only launches a local process on 127.0.0.1.
# =============================================================================

[CmdletBinding()]
param(
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot 'edge-config.ps1')

# --- repo root + config ------------------------------------------------------
$RepoRoot = Get-EdgeRepoRoot -ScriptPath $PSCommandPath
$ConfigPath = Join-Path $RepoRoot 'config\edge-production.local.yaml'
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Local edge config missing: $ConfigPath`nCopy config\edge-production.example.yaml to edge-production.local.yaml and fill in the real values first."
}

$cfg = Read-EdgeConfigFile -Path $ConfigPath

if (-not $cfg.Contains('proxy') -or -not $cfg.Contains('runtime')) {
    throw "Edge config must contain 'proxy:' and 'runtime:' sections."
}

$Port = [int]$cfg['proxy']['port']
$PidFile = [string]$cfg['runtime']['pid_file']
$PidFile = [System.Environment]::ExpandEnvironmentVariables($PidFile)

# --- refuse duplicates (same guards as the inner script) ---------------------
$existing = Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $owner = Get-Process -Id $existing[0].OwningProcess -ErrorAction SilentlyContinue
    throw "Port $Port is already listening (PID $($owner.Id) $($owner.ProcessName)). Refusing to start a duplicate edge instance."
}
if (Test-Path -LiteralPath $PidFile) {
    $pidLine = (Get-Content -LiteralPath $PidFile -TotalCount 1 | Select-Object -First 1)
    if ($pidLine -match '^\d+$') {
        $alive = Get-Process -Id ([int]$pidLine) -ErrorAction SilentlyContinue
        if ($alive) {
            throw "PID file $PidFile points at a live process ($pidLine). Refusing to start a duplicate edge instance."
        }
    }
    Write-Warning "Removing stale PID file: $PidFile"
    Remove-Item -LiteralPath $PidFile -Force
}

# --- the inner (regular) start script runs everything real -------------------
$InnerScript = Join-Path $PSScriptRoot 'Start-LiteLLM-Edge.ps1'
if (-not (Test-Path -LiteralPath $InnerScript)) {
    throw "Inner launcher not found: $InnerScript"
}

$pwshExe = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwshExe) {
    $pwshExe = (Get-Command powershell -ErrorAction SilentlyContinue).Source
}
if (-not $pwshExe) {
    throw "Neither pwsh nor powershell found on PATH."
}

Write-Host '=== LiteLLM Differential Context edge - DETACHED launch ==='
Write-Host ("config        : " + $ConfigPath)
Write-Host ("inner launcher: " + $InnerScript)
Write-Host ("bind          : 127.0.0.1:${Port} (loopback only)")
Write-Host ("pid file      : " + $PidFile)
Write-Host ("shell         : " + $pwshExe)
Write-Host 'The edge will run in its own hidden console. Closing THIS window'
Write-Host 'will NOT stop the edge. Stop it with scripts\Stop-LiteLLM-Edge.ps1.'

if ($DryRun) {
    Write-Host ''
    Write-Host 'DRY RUN - no process launched. Launch would use:'
    Write-Host ("  Start-Process -FilePath '" + $pwshExe + "' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','" + $InnerScript + "' -WindowStyle Hidden")
    Write-Host 'Nothing was started. SGLang and remote routing are untouched.'
    exit 0
}

# --- launch: hidden, detached console ----------------------------------------
$inner = Start-Process -FilePath $pwshExe `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $InnerScript) `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -PassThru

Write-Host ''
Write-Host ("Detached supervisor started (PID " + $inner.Id + "). Waiting for the edge to come up...")

# --- liveness / readiness probe (bounded) ------------------------------------
function Test-LocalEndpoint {
    param([string]$Url)
    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        return $resp.StatusCode
    } catch {
        return $null
    }
}

$liveOk = $false
for ($i = 0; $i -lt 60; $i++) {
    if ($inner.HasExited) {
        throw "Detached supervisor exited during startup (exit code $($inner.ExitCode)). See D:\_SATIN_AI_2\logs\litellm-edge\litellm-edge.err.log"
    }
    $code = Test-LocalEndpoint -Url "http://127.0.0.1:${Port}/health/liveliness"
    if ($code -eq 200) { $liveOk = $true; break }
    Start-Sleep -Seconds 2
}

if (-not $liveOk) {
    Write-Warning "Liveness endpoint not answering within 120s (supervisor still running, PID $($inner.Id)). Check the edge err log."
    exit 1
}

Write-Host ("liveness  : http://127.0.0.1:${Port}/health/liveliness -> 200")
$readyOk = $false
for ($i = 0; $i -lt 30; $i++) {
    $code = Test-LocalEndpoint -Url "http://127.0.0.1:${Port}/health/readiness"
    if ($code -eq 200) { $readyOk = $true; break }
    Start-Sleep -Seconds 2
}
if ($readyOk) {
    Write-Host ("readiness : http://127.0.0.1:${Port}/health/readiness -> 200")
} else {
    Write-Warning "Readiness endpoint not answering within 60s after liveness. Check the edge err log."
    exit 1
}

Write-Host ''
Write-Host ("Edge is UP and DETACHED (supervisor PID " + $inner.Id + ").")
Write-Host 'Closing this window will not stop it. Stop it with scripts\Stop-LiteLLM-Edge.ps1.'
Write-Host 'SGLang and remote routing were NOT touched.'
exit 0