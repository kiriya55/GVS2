@echo off
chcp 65001 >nul 2>&1
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe app.py %*
