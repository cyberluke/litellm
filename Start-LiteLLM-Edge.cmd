@echo off
rem ===========================================================================
rem Start-LiteLLM-Edge.cmd - double-click / one obvious file launcher.
rem Invokes the PowerShell launcher cleanly. Nothing is started by this file
rem itself; all logic lives in scripts\Start-LiteLLM-Edge.ps1.
rem ===========================================================================
setlocal
where pwsh >nul 2>nul
if %errorlevel%==0 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-LiteLLM-Edge.ps1" %*
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-LiteLLM-Edge.ps1" %*
)
set EXITCODE=%errorlevel%
endlocal & exit /b %EXITCODE%