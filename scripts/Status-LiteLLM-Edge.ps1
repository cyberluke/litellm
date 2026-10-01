# =============================================================================
# Status-LiteLLM-Edge.ps1 — report the local Differential Context edge state.
#
# Displays: process state, local health/readiness, local port, edge DB path,
# remote capability status (best effort), negotiated transport info (best
# effort from the local /metrics surface). Read-only - never starts or stops
# anything, never touches SGLang or remote routing.
# =============================================================================

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot 'edge-config.ps1')

$RepoRoot = Get-EdgeRepoRoot -ScriptPath $PSCommandPath
$ConfigPath = Join-Path $RepoRoot 'config\edge-production.local.yaml'

Write-Host '=== LiteLLM Differential Context edge - status ==='

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Write-Warning "Config missing: $ConfigPath"
    exit 1
}
$cfg = Read-EdgeConfigFile -Path $ConfigPath

$BindHost = [string]$cfg['proxy']['host']
$Port = [int]$cfg['proxy']['port']
$LocalBaseUrl = "http://${Host}:${Port}"

$PidFile = [string]$cfg['runtime']['pid_file']
$PidFile = [System.Environment]::ExpandEnvironmentVariables($PidFile)
$LogDir = [string]$cfg['runtime']['log_dir']
$LogDir = [System.Environment]::ExpandEnvironmentVariables($LogDir)

$envBlock = $cfg['env']
$DbPath = [string]$envBlock['EDGE_STATE_DB_PATH']
if ($DbPath -match '\$\{(?<v>[A-Za-z0-9_]+)\}') {
    $DbPath = [regex]::Replace($DbPath, '\$\{(?<v>[A-Za-z0-9_]+)\}', {
        param($m)
        $val = [System.Environment]::GetEnvironmentVariable($m.Groups['v'].Value)
        if ($null -eq $val) { $m.Value } else { $val }
    })
}
$BaseUrl = [string]$envBlock['EDGE_SSEPROXY_BASE_URL']
if ($BaseUrl -match '^\$\{(?<v>[A-Za-z0-9_]+)\}$') {
    $BaseUrl = [System.Environment]::GetEnvironmentVariable($matches['v'])
}

# --- process state -----------------------------------------------------------
$edgePid = $null
if (Test-Path -LiteralPath $PidFile) {
    $pidLine = Get-Content -LiteralPath $PidFile -TotalCount 1 | Select-Object -First 1
    if ($pidLine -match '^\d+$') {
        $edgePid = [int]$pidLine
    }
}

$listeners = @(Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($edgePid) {
    $p = Get-Process -Id $edgePid -ErrorAction SilentlyContinue
    if ($p) {
        Write-Host ("process    : RUNNING (PID " + $edgePid + ", " + $p.ProcessName + ", started " + $p.StartTime.ToString('s') + ")")
    } else {
        Write-Host 'process    : NOT RUNNING (stale PID file present)'
    }
} elseif ($listeners.Count -gt 0) {
    Write-Host ("process    : LISTENING but no PID file (PID " + $listeners[0].OwningProcess + ")")
    $edgePid = $listeners[0].OwningProcess
} else {
    Write-Host 'process    : NOT RUNNING'
}

Write-Host ("local port : " + $BindHost + ":" + $Port)
Write-Host ("edge DB    : " + $DbPath)
Write-Host ("logs       : " + $LogDir)
Write-Host ("config     : " + $ConfigPath)

# --- local health --------------------------------------------------------------
function Test-LocalEndpoint {
    param([string]$Url)
    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        return $resp.StatusCode
    } catch {
        return $null
    }
}
$live = Test-LocalEndpoint -Url "$LocalBaseUrl/health/liveliness"
$ready = Test-LocalEndpoint -Url "$LocalBaseUrl/health/readiness"
Write-Host ("liveness   : " + $(if ($live) { $live } else { 'no response' }))
Write-Host ("readiness  : " + $(if ($ready) { $ready } else { 'no response' }))

# --- remote capabilities (best effort) ------------------------------------------
if ($BaseUrl -and $BaseUrl -notlike '*CHANGE_ME*' -and $BaseUrl -notlike '${*') {
    try {
        $cap = Invoke-WebRequest -Uri "$BaseUrl/v1/transport/capabilities" -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
        $capJson = $cap.Content | ConvertFrom-Json
        Write-Host ("remote caps: " + $BaseUrl + " -> " + $cap.StatusCode + " (edge " + $capJson.edge_protocol_version + ", wan " + ($capJson.wan_http_versions -join ',') + ")")
    } catch {
        Write-Host ("remote caps: UNREACHABLE (" + $_.Exception.Message + ")")
    }
} else {
    Write-Host 'remote caps: not configured (placeholder base URL)'
}

# --- negotiated transport (best effort from local /metrics) -----------------------
$metrics = $null
try {
    $metrics = (Invoke-WebRequest -Uri "$LocalBaseUrl/metrics" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop).Content
} catch { }
if ($metrics) {
    $h2 = [regex]::Match($metrics, 'edge_http_negotiated_version_total\{[^}]*http_version="2"[^}]*\}\s+(\d+)')
    $h1 = [regex]::Match($metrics, 'edge_http_negotiated_version_total\{[^}]*http_version="1\.1"[^}]*\}\s+(\d+)')
    $ops = [regex]::Matches($metrics, 'edge_dc_operation_total\{[^}]*\}') | ForEach-Object { $_.Value }
    if ($h2.Success -or $h1.Success -or $ops.Count -gt 0) {
        Write-Host ("transport  : HTTP/2 requests=" + $(if ($h2.Success) { $h2.Groups[1].Value } else { '0' }) + ", HTTP/1.1=" + $(if ($h1.Success) { $h1.Groups[1].Value } else { '0' }))
        Write-Host ("dc ops     : " + $ops.Count + " metric series exposed")
    } else {
        Write-Host 'transport  : metrics surface exposed but no edge transport series yet (no edge request served)'
    }
} else {
    Write-Host 'transport  : /metrics not reachable'
}

exit 0