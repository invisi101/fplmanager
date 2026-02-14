#!/bin/bash
cd /home/neil/dev/fpl-manager
lsof -ti:9875 | xargs kill -9 2>/dev/null
sleep 0.5
.venv/bin/python -m src.app &
sleep 2
xdg-open http://127.0.0.1:9875
