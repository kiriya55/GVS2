#!/usr/bin/env bash
set -e

if [ ! -f ".venv/bin/python" ]; then
    echo "[!] Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

.venv/bin/python app.py "$@"
