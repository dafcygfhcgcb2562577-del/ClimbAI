@echo off
chcp 65001 >nul
title ClimbAI - LAN link
cd /d "%~dp0"

echo.
echo ========================================
echo   ClimbAI - LAN (odin Wi-Fi)
echo ========================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\open_firewall.ps1"

set CLIMB_API_HOST=0.0.0.0
set CLIMB_OPEN_BROWSER=0

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
  taskkill /F /PID %%P >nul 2>&1
)

echo Starting server on 0.0.0.0:8000 ...
start "ClimbAI Server" cmd /k "cd /d "%~dp0" && set CLIMB_API_HOST=0.0.0.0 && set CLIMB_OPEN_BROWSER=0 && run_all_in_one.bat"

echo Waiting for server...
timeout /t 15 /nobreak >nul

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\show_lan_link.ps1"

echo.
pause
