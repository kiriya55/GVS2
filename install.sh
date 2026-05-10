#!/usr/bin/env bash
set -e

echo "[*] Creating virtual environment..."
python3 -m venv .venv

echo "[*] Installing dependencies..."
.venv/bin/pip install -r requirements.txt

echo "[+] Done. You can now run ./app.sh or ./benchmark.sh to start the application."
