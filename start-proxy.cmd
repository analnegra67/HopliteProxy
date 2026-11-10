@echo off
rem Starts the Hoplite OpenAI proxy on http://127.0.0.1:8000/v1. Close this window to stop it.
title Hoplite OpenAI proxy
cd /d "%~dp0"
set "PY=%USERPROFILE%\.venvs\hoplite-proxy\Scripts\python.exe"
if not exist "%PY%" (
  echo Python venv not found: %PY%
  echo Create it as described in README.md.
  pause
  exit /b 1
)
"%PY%" -m app
echo.
echo The proxy has stopped. Press any key to close this window.
pause >nul
