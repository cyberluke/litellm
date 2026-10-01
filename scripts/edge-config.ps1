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

        # strip comments outside values (only full-line and trailing comments
        # preceded by whitespace are honored)
        $stripped = $line -replace '^\s*#.*$', ''
        if ($stripped -match '^(?<indent>\s*)(?<name>[A-Za-z0-9_]+):\s*(?<value>.*)$') {
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
    # ${VAR} -> process environment when set; otherwise keep the literal.
    if ($Value -match '^\$\{(?<var>[A-Za-z0-9_]+)\}$') {
        $var = $matches['var']
        $envValue = [System.Environment]::GetEnvironmentVariable($var)
        if ($null -ne $envValue -and $envValue -ne '') {
            return $envValue
        }
        return $Value  # keep the ${VAR} marker; caller decides how to report
    }
    return $Value
}

function Export-EdgeEnvBlock {
    param([System.Collections.IDictionary]$EnvBlock)
    foreach ($key in $EnvBlock.Keys) {
        [System.Environment]::SetEnvironmentVariable([string]$key, [string]$EnvBlock[$key], 'Process')
    }
}

# NOTE: this file is dot-sourced (not imported as a module), so all
# functions above are available in the caller's scope automatically.