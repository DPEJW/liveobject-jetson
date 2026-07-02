#!/usr/bin/env bash
# Launch the liveobject detection web app.
cd "$(dirname "$0")" || exit 1
# Prefer the project venv (the Blackwell/GB10 stack lives here) if present.
if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python app.py
fi
exec python3 app.py
