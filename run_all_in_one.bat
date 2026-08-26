@echo off
title ClimbAI
cd /d "%~dp0"
set "PY=%~dp0venv\Scripts\python.exe"
if not defined CLIMB_OPEN_BROWSER set CLIMB_OPEN_BROWSER=1

where python >nul 2>&1 || goto :fail
if not exist "%PY%" python -m venv "%~dp0venv" >nul 2>&1
if not exist "%PY%" goto :fail

if not exist "%~dp0artifacts" mkdir "%~dp0artifacts" >nul 2>&1
if not exist "%~dp0artifacts\web_jobs" mkdir "%~dp0artifacts\web_jobs" >nul 2>&1

if not exist "%~dp0artifacts\.deps_ok" (
  "%PY%" -c "import fastapi, uvicorn, cv2, mediapipe" >nul 2>&1
  if errorlevel 1 (
    echo Установка зависимостей...
    "%PY%" -m pip install -q --default-timeout=120 -r "%~dp0requirements-web.txt"
    if errorlevel 1 goto :fail
  )
  echo ok>"%~dp0artifacts\.deps_ok"
)


rem Старый сервер держит в памяти старую версию кода: Python читает файлы один
rem раз, при запуске. Если его не остановить, правки не применятся, и это
rem выглядит как «ничего не поменялось». Убиваем и по порту, и по имени
rem скрипта: сервер мог быть запущен другим Python, не из venv.
echo Останавливаю прошлый сервер...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
  taskkill /F /PID %%P >nul 2>&1
)
wmic process where "name like 'python%%' and commandline like '%%run_backend%%'" call terminate >nul 2>&1
timeout /t 1 /nobreak >nul

"%PY%" "%~dp0run_backend.py"
if errorlevel 1 echo Ошибка запуска. Порт 8000 занят?
pause
exit /b %ERRORLEVEL%

:fail
echo Нужен Python. Установите Python 3.10+ и запустите снова.
pause
exit /b 1
