"""Linux launcher — kills stale process on port 9875, starts Flask server, opens browser."""

import os
import signal
import subprocess
import threading
import time
import webbrowser

URL = "http://127.0.0.1:9875"
PORT = 9875


def kill_stale_process():
    """Kill any existing process listening on the server port."""
    try:
        result = subprocess.run(
            ["fuser", f"{PORT}/tcp"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            pid = pid.strip()
            if pid.isdigit():
                print(f"Killing stale process {pid} on port {PORT}")
                os.kill(int(pid), signal.SIGKILL)
        if pids:
            time.sleep(0.5)
    except FileNotFoundError:
        # fuser not available, try lsof as fallback
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{PORT}"],
                capture_output=True, text=True
            )
            for pid in result.stdout.strip().splitlines():
                pid = pid.strip()
                if pid.isdigit():
                    print(f"Killing stale process {pid} on port {PORT}")
                    os.kill(int(pid), signal.SIGKILL)
            if result.stdout.strip():
                time.sleep(0.5)
        except FileNotFoundError:
            pass


def open_browser():
    """Wait for the server to start, then open the browser."""
    time.sleep(1.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    kill_stale_process()

    print(f"Starting Gaffer at {URL}")
    print("Press Ctrl+C to stop the server.\n")

    threading.Thread(target=open_browser, daemon=True).start()

    from src.app import app

    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
