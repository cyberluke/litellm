# =============================================================================
# edge-config.ps1 — shared config loader for the Differential Context edge
# launcher scripts (Start/Stop/Status-LiteLLM-Edge.ps1).
#
# Reads the flat YAML convention used by config/edge-production.local.yaml:
#   - sections are `name:` lines at column 0;
#   - entries are `NAME: value` lines inside a section;
#   - a value of the form "${VAR}" is resolved from the process environment
#     when that variable is set, otherwise the literal string is kept.
# No YAML library is required; this parser only accepts the documented flat
# subset and fails loudly on anything else.
# =============================================================================

Set-StrictMode -Version Latest

function Get-EdgeRepoRoot {
    param([string]$ScriptPath)
    # scripts/edge-config.ps1 -> repo root is two levels up.
    $dir = Split-Path -Parent (Split-Path -Parent $ScriptPath)
    return $dir
}

function Read-EdgeConfigFile {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Edge config not found: $Path"
    }

    $sections = [ordered]@{}
    $current = $null
    $lineNo = 0

    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $lineNo++
        $line = $rawLine.TrimEnd()

        # strip full-line comments first
        $stripped = $line -replace '^\s*#.*$', ''
        if ($stripped -match '^(?<indent>\s*)(?<name>[A-Za-z0-9_]+):\s*(?<value>[^#]*?)\s*(#.*)?$') {
            $indent = $matches['indent']
            $name = $matches['name']
            $value = $matches['value']
            if ($indent.Length -eq 0) {
                # section header
                $current = $name
                if (-not $sections.Contains($name)) {
                    $sections[$name] = [ordered]@{}
                }
                continue
            }
            if ($null -eq $current) {
                throw "Entry '$name' appears before any section header (line $lineNo of $Path)"
            }
            $value = $value.Trim()
            # strip surrounding quotes
            if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $sections[$current][$name] = $value
            continue
        }
        $trimmed = $line.Trim()
        if ($trimmed -eq '' -or $trimmed.StartsWith('#')) {
            continue
        }
        throw "Unparseable line $lineNo in $Path : $line"
    }
    return $sections
}

function Resolve-EdgeEnvValue {
    param([string]$Value, [string]$Name)
    # ${VAR} -> process environment when set; expands EVERY occurrence
    # (also suffixed forms like "${LOCALAPPDATA}/VIVERRA/..."); keeps the
    # literal ${VAR} text when the variable is unset.
    return [regex]::Replace($Value, '\$\{(?<var>[A-Za-z0-9_]+)\}', {
        param($m)
        $envValue = [System.Environment]::GetEnvironmentVariable($m.Groups['var'].Value)
        if ($null -ne $envValue -and $envValue -ne '') {
            return $envValue
        }
        return $m.Value
    })
}

function Export-EdgeEnvBlock {
    param([System.Collections.IDictionary]$EnvBlock)
    foreach ($key in $EnvBlock.Keys) {
        [System.Environment]::SetEnvironmentVariable([string]$key, [string]$EnvBlock[$key], 'Process')
    }
}

function Invoke-EdgeWebRequest {
    # WAN probe helper: the bench Caddy frontend serves a self-signed cert
    # (the edge transport itself runs with EDGE_SSEPROXY_VERIFY_TLS=false),
    # so read-only probes must tolerate it. PS7 has -SkipCertificateCheck;
    # Windows PowerShell 5.1 falls back to the ServicePointManager callback.
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [System.Collections.IDictionary]$Headers = @{},
        [int]$TimeoutSec = 10
    )
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        return Invoke-WebRequest -Uri $Uri -Headers $Headers -UseBasicParsing -SkipCertificateCheck -TimeoutSec $TimeoutSec -ErrorAction Stop
    }
    $prev = [System.Net.ServicePointManager]::ServerCertificateValidationCallback
    try {
        [System.Net.ServicePointManager]::ServerCertificateValidationCallback = { $true }
        return Invoke-WebRequest -Uri $Uri -Headers $Headers -UseBasicParsing -TimeoutSec $TimeoutSec -ErrorAction Stop
    } finally {
        [System.Net.ServicePointManager]::ServerCertificateValidationCallback = $prev
    }
}

# NOTE: this file is dot-sourced (not imported as a module), so all
# functions above are available in the caller's scope automatically.