@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    echo Python virtual environment not found.
    echo Please run install.sh or create the venv first.
    pause
    exit /b 1
)

set "PRODUCTION_MODE=true"
start "AI Video Transcriber" cmd /k "%PYTHON_EXE% start.py --prod"

timeout /t 5 /nobreak >nul
start "" http://localhost:8000

echo Server starting...
echo Opened http://localhost:8000
pause
