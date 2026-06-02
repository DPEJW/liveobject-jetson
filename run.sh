#!/usr/bin/env bash
# Launch the liveobject detection web app.
cd "$(dirname "$0")" || exit 1
exec python3 app.py
