"""PyInstaller entry point — starts Flask server and opens browser."""

import multiprocessing
import sys
import threading
import time
import traceback
import webbrowser

URL = "http://127.0.0.1:9875"


def open_browser():
    """Wait briefly for the server to start, then open the browser."""
    time.sleep(1.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        print(f"Starting FPL Points Predictor at {URL}")
        print("Close this window to stop the server.\n")

        threading.Thread(target=open_browser, daemon=True).start()

        from src.app import app

        app.run(host="127.0.0.1", port=9875, debug=False, threaded=True)
    except Exception:
        traceback.print_exc()
        input("\nPress Enter to exit...")
