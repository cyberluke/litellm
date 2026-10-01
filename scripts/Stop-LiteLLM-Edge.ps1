# =============================================================================
# Stop-LiteLLM-Edge.ps1 — stop the local Differential Context edge process.
#
# Reads the PID file, verifies the process really is the LiteLLM edge launch
# (command line match), requests graceful termination first, waits a bounded
# window, forces only if the graceful stop fails, then removes the PID file.
# Never touches SGLang or remote routing.
# =============================================================================

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot 'edge-config.ps1')

$RepoRoot = Get-EdgeRepoRoot -ScriptPath $PSCommandPath
$ConfigPath = Join-Path $RepoRoot 'config\edge-production.local.yaml'
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Write-Warning "Config missing ($ConfigPath) - using default PID file location."
    $PidFile = Join-Path $RepoRoot 'litellm-edge.pid'
} else {
    $cfg = Read-EdgeConfigFile -Path $ConfigPath
    $PidFile = [string]$cfg['runtime']['pid_file']
    $PidFile = [System.Environment]::ExpandEnvironmentVariables($PidFile)
}

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host "No PID file at $PidFile - the edge does not appear to be running."
    exit 0
}

$lines = @(Get-Content -LiteralPath $PidFile)
if ($lines.Count -lt 1 -or $lines[0] -notmatch '^\d+$') {
    Write-Warning "PID file $PidFile is malformed - removing it."
    Remove-Item -LiteralPath $PidFile -Force
    exit 0
}

$targetPid = [int]$lines[0]
$recordedStart = if ($lines.Count -ge 2) { $lines[1] } else { '' }
$recordedCmd = if ($lines.Count -ge 3) { $lines[2] } else { '' }

$proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
if (-not $proc) {
    Write-Host "PID $targetPid is not running - removing stale PID file."
    Remove-Item -LiteralPath $PidFile -Force
    exit 0
}

# --- verify this process belongs to this edge launch -------------------------
$cmdLine = ''
try {
    $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $targetPid").CommandLine
} catch { }
if ($cmdLine -notmatch 'litellm\.proxy\.proxy_cli') {
    Write-Warning "PID $targetPid ($($proc.ProcessName)) is NOT a LiteLLM edge launch - NOT stopping it."
    Write-Warning "Command line: $cmdLine"
    exit 1
}
Write-Host "Verified edge process: PID $targetPid ($($proc.ProcessName))"

# --- graceful first, bounded wait, force fallback -----------------------------
$graceful = $proc.CloseMainWindow()
if ($graceful) {
    Write-Host 'Graceful termination requested (CloseMainWindow).'
} else {
    Write-Warning 'Graceful termination not available for this process (no window) - falling back to Stop-Process after a short wait.'
}

$deadline = (Get-Date).AddSeconds(15)
while ((Get-Date) -lt $deadline -and -not $proc.HasExited) {
    Start-Sleep -Milliseconds 500
}

if (-not $proc.HasExited) {
    Write-Warning "Process still alive after graceful window - forcing stop (PID $targetPid)."
    Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
    $proc.WaitForExit(10000) | Out-Null
}

if ($proc.HasExited) {
    Write-Host ("Edge stopped (PID " + $targetPid + ", exit code " + $proc.ExitCode + ").")
} else {
    Write-Warning "Edge process (PID $targetPid) could not be stopped. Check for a hung process."
    exit 1
}

Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
Write-Host 'PID file removed. SGLang and remote routing were NOT touched.'
exit 0