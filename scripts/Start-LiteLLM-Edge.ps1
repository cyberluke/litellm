# =============================================================================
# Start-LiteLLM-Edge.ps1 — start the local Differential Context edge
# (LiteLLM proxy with the differential_sseproxy edge profile).
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
$ExamplePath = Join-Path $RepoRoot 'config\edge-production.example.yaml'

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Local edge config missing: $ConfigPath`nCopy config\edge-production.example.yaml to edge-production.local.yaml and fill in the real values first."
}

$cfg = Read-EdgeConfigFile -Path $ConfigPath

if (-not $cfg.Contains('env') -or -not $cfg.Contains('proxy') -or -not $cfg.Contains('runtime')) {
    throw "Edge config must contain 'env:', 'proxy:' and 'runtime:' sections."
}

# --- resolve env block (${VAR} indirection) ---------------------------------
$resolvedEnv = [ordered]@{}
foreach ($key in $cfg['env'].Keys) {
    $raw = [string]$cfg['env'][$key]
    $resolvedEnv[$key] = Resolve-EdgeEnvValue -Value $raw -Name $key
}

# Secrets must be present when the edge profile is enabled.
$edgeEnabled = [string]$resolvedEnv['EDGE_SSEPROXY_ENABLED']
$isEnabled = ($edgeEnabled -eq '1' -or $edgeEnabled -eq 'true' -or $edgeEnabled -eq 'yes')
if ($isEnabled) {
    $apiKey = [string]$resolvedEnv['EDGE_SSEPROXY_API_KEY']
    if ($apiKey -eq '' -or $apiKey -like '${*') {
        throw "EDGE_SSEPROXY_API_KEY is not set. Set the EDGE_SSEPROXY_API_KEY environment variable (or fill edge-production.local.yaml) before starting the edge profile."
    }
    $baseUrl = [string]$resolvedEnv['EDGE_SSEPROXY_BASE_URL']
    if ($baseUrl -eq '' -or $baseUrl -like '${*' -or $baseUrl -like '*CHANGE_ME*') {
        Write-Warning "EDGE_SSEPROXY_BASE_URL is a placeholder ($baseUrl) - the edge will not reach the WAN frontend until the real Caddy host is configured."
    }
    if ($baseUrl -like 'http://*' -or $baseUrl -match 'localhost') {
        Write-Warning "EDGE_SSEPROXY_BASE_URL ($baseUrl) does not look like the production WAN TLS frontend - verify before cutover."
    }
}

# --- proxy / runtime settings ------------------------------------------------
$BindHost = [string]$cfg['proxy']['host']
if ($BindHost -ne '127.0.0.1' -and $BindHost -ne 'localhost') {
    throw "Edge profile must bind loopback only; proxy.host is '$BindHost'."
}
$Port = [int]$cfg['proxy']['port']
$NumWorkers = [int]$cfg['proxy']['num_workers']
if ($NumWorkers -ne 1) {
    throw "Edge profile requires a single worker (proxy.num_workers must be 1); the worker_guard fails startup otherwise."
}

$pythonRel = [string]$cfg['runtime']['python_env']
$Python = if ([System.IO.Path]::IsPathRooted($pythonRel)) { $pythonRel } else { Join-Path $RepoRoot $pythonRel }
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Dedicated Python environment not found: $Python`nCreate it first, e.g.: uv venv --python 3.13 .venv && uv pip install --python .venv\Scripts\python.exe `"litellm[proxy]`" zstandard && uv pip install --python .venv\Scripts\python.exe -e D:\_SATIN_AI_2\differential-context`n(Editable install of the fork source needs rustc >= 1.94 for the litellm-rust bridge; the wheel-based env runs the LOCAL fork source from the repo root.)"
}

$LogDir = [string]$cfg['runtime']['log_dir']
$LogDir = [System.Environment]::ExpandEnvironmentVariables($LogDir)
if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}
$LogOut = Join-Path $LogDir 'litellm-edge.out.log'
$LogErr = Join-Path $LogDir 'litellm-edge.err.log'
$PidFile = [string]$cfg['runtime']['pid_file']
$PidFile = [System.Environment]::ExpandEnvironmentVariables($PidFile)
$ProbeRemote = [bool]$cfg['runtime']['probe_remote_capabilities']

$DbPath = [string]$resolvedEnv['EDGE_STATE_DB_PATH']
if ($DbPath -eq '' -or $DbPath -like '${*') {
    $DbPath = [System.Environment]::GetEnvironmentVariable('LOCALAPPDATA') + '\VIVERRA\LiteLLM\edge-state.sqlite3'
}

$BaseUrl = [string]$resolvedEnv['EDGE_SSEPROXY_BASE_URL']
$LocalBaseUrl = "http://${BindHost}:${Port}"

# --- refuse duplicates -------------------------------------------------------
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

# --- summary -----------------------------------------------------------------
Write-Host '=== LiteLLM Differential Context edge - launch summary ==='
Write-Host ("config        : " + $ConfigPath)
Write-Host ("python env    : " + $Python)
Write-Host ("bind          : ${BindHost}:${Port} (loopback only)")
Write-Host ("workers       : " + $NumWorkers)
Write-Host ("edge sqlite   : " + $DbPath)
Write-Host ("logs          : " + $LogOut + " / " + $LogErr)
Write-Host ("WAN base URL  : " + $BaseUrl)

if ($DryRun) {
    Write-Host ''
    Write-Host 'DRY RUN - no process launched. Launch would use:'
    Write-Host ("  & '" + $Python + "' -m litellm.proxy.proxy_cli --host " + $BindHost + " --port " + $Port + " --num_workers " + $NumWorkers)
    Write-Host 'Nothing was started. SGLang and remote routing are untouched.'
    exit 0
}

# --- export edge env + launch -------------------------------------------------
Export-EdgeEnvBlock -EnvBlock $resolvedEnv

# UTF-8 stdio for the child: LiteLLM's startup banner contains non-cp1252
# characters and crashes the proxy on a cp1252 console (UnicodeEncodeError,
# exit code 3) without this.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $Python
$psi.WorkingDirectory = $RepoRoot
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
foreach ($arg in @('-m', 'litellm.proxy.proxy_cli', '--host', $BindHost, '--port', [string]$Port, '--num_workers', [string]$NumWorkers)) {
    [void]$psi.ArgumentList.Add($arg)
}

$proc = [System.Diagnostics.Process]::new()
$proc.StartInfo = $psi

# append-only log pump (never truncate previous diagnostics). Writes every
# line IMMEDIATELY (AutoFlush) so logs survive regardless of how/when the
# launcher or the child exits. PS 7.6-rc/.NET 10 exposes Process output
# events only through Register-ObjectEvent (the property adapter `+=` form
# fails with "property not found"), so the writer is carried via MessageData.
$outWriter = [System.IO.StreamWriter]::new($LogOut, $true)
$errWriter = [System.IO.StreamWriter]::new($LogErr, $true)
$outWriter.AutoFlush = $true
$errWriter.AutoFlush = $true

$outAction = {
    $writer = $Event.MessageData
    $data = $Event.SourceEventArgs.Data
    if ($null -ne $data) { try { $writer.WriteLine($data) } catch { } }
}
$errAction = {
    $writer = $Event.MessageData
    $data = $Event.SourceEventArgs.Data
    if ($null -ne $data) { try { $writer.WriteLine($data) } catch { } }
}
$null = Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -MessageData $outWriter -Action $outAction
$null = Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived -MessageData $errWriter -Action $errAction

$proc.Start() | Out-Null
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

$started = Get-Date
$pidContent = @(
    [string]$proc.Id,
    $started.ToString('o'),
    ($psi.FileName + ' ' + ($psi.ArgumentList -join ' ')),
    [string]$Port
) -join "`n"
Set-Content -LiteralPath $PidFile -Value $pidContent -Encoding UTF8

Write-Host ''
Write-Host ("PID           : " + $proc.Id)
Write-Host ("local base URL: " + $LocalBaseUrl + "/v1")
Write-Host ("pid file      : " + $PidFile)

# --- liveness / readiness -----------------------------------------------------
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
    if ($proc.HasExited) {
        throw "Edge process exited during startup (exit code $($proc.ExitCode)). See $LogErr"
    }
    $code = Test-LocalEndpoint -Url "$LocalBaseUrl/health/liveliness"
    if ($code -eq 200) { $liveOk = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $liveOk) {
    Write-Warning "Liveness endpoint not answering within 120s (process still running, PID $($proc.Id)). Readiness may still be pending."
} else {
    Write-Host ("liveness  : " + $LocalBaseUrl + "/health/liveliness -> 200")
    $readyOk = $false
    for ($i = 0; $i -lt 30; $i++) {
        $code = Test-LocalEndpoint -Url "$LocalBaseUrl/health/readiness"
        if ($code -eq 200) { $readyOk = $true; break }
        Start-Sleep -Seconds 2
    }
    if ($readyOk) {
        Write-Host ("readiness : " + $LocalBaseUrl + "/health/readiness -> 200")
    } else {
        Write-Warning "Readiness endpoint not answering within 60s after liveness. Check $LogErr"
    }
}

# --- optional remote capabilities probe ---------------------------------------
if ($ProbeRemote -and $BaseUrl -notlike '*CHANGE_ME*' -and $BaseUrl -notlike '${*') {
    try {
        $headers = @{ Authorization = 'Bearer ' + [string]$resolvedEnv['EDGE_SSEPROXY_API_KEY'] }
        $cap = Invoke-EdgeWebRequest -Uri "$BaseUrl/v1/transport/capabilities" -Headers $headers -TimeoutSec 10
        Write-Host ("remote capabilities: " + $BaseUrl + "/v1/transport/capabilities -> " + $cap.StatusCode)
    } catch {
        Write-Warning ("remote capabilities: UNREACHABLE - " + $BaseUrl + "/v1/transport/capabilities (" + $_.Exception.Message + ")")
    }
} else {
    Write-Host 'remote capabilities: skipped (placeholder base URL or probe disabled)'
}

# --- supervisor: stay attached to the edge process ------------------------------
Write-Host ''
Write-Host ("Edge running (PID " + $proc.Id + "). Keep this window open; live logs are appended to:")
Write-Host ("  " + $LogOut)
Write-Host ("  " + $LogErr)
Write-Host 'Stop it with Ctrl+C here or with scripts\Stop-LiteLLM-Edge.ps1 (another window).'
Write-Host 'SGLang and remote routing were NOT touched.'
try {
    $proc.WaitForExit()
    Write-Host ''
    Write-Host ("Edge process exited (code " + $proc.ExitCode + "). See the logs above.")
} finally {
    try { $outWriter.Dispose() } catch { }
    try { $errWriter.Dispose() } catch { }
}
exit 0