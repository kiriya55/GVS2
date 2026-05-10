@echo off
chcp 65001 >nul 2>&1
setlocal

echo [*] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo [!] Failed to create virtual environment. Please ensure Python 3.10+ is installed and added to PATH.
    pause
    exit /b 1
)

echo [*] Installing dependencies...
.venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo [!] Failed to install dependencies.
    pause
    exit /b 1
)

echo [+] Done. You can now run app.bat or benchmark.bat to start the application.
pause
