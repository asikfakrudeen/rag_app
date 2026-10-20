"""
main.py — Single entry point for the Legal Contract Review Assistant.

Launches both services with one command:
  python main.py

Services started:
  - FastAPI backend  → http://localhost:8000
  - Streamlit UI     → http://localhost:8501
"""

import os
import sys
import subprocess
import signal
import time

FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║        ⚖️  Legal Contract Review Assistant — Week 12         ║
║                   Track F · Capstone Build                   ║
╚══════════════════════════════════════════════════════════════╝
"""


def check_env():
    """Abort early if the API key is missing."""
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key or key.startswith("AIzaSy..."):
        print("❌  GOOGLE_API_KEY is missing or still set to the placeholder.")
        print("    Add a real key to your .env file and try again.")
        sys.exit(1)
    print("✅  GOOGLE_API_KEY found.")


def resolve_python():
    """Return the Python/uvicorn executable from the venv if present."""
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    venv_uvicorn = os.path.join(".venv", "Scripts", "uvicorn.exe")
    if os.path.isfile(venv_python):
        return venv_python, venv_uvicorn
    return sys.executable, "uvicorn"


def start_fastapi(uvicorn_exe):
    """Start the FastAPI backend in a subprocess."""
    cmd = [
        uvicorn_exe,
        "api:app",
        "--host", FASTAPI_HOST,
        "--port", str(FASTAPI_PORT),
        "--reload",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def start_streamlit(python_exe):
    """Start the Streamlit UI in a subprocess."""
    cmd = [
        python_exe, "-m", "streamlit", "run", "app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main():
    print(BANNER)
    check_env()

    python_exe, uvicorn_exe = resolve_python()

    print("🚀  Starting FastAPI backend ...")
    api_proc = start_fastapi(uvicorn_exe)

    print("🚀  Starting Streamlit UI ...")
    ui_proc = start_streamlit(python_exe)

    # Give processes a moment to spin up
    time.sleep(3)

    print()
    print("══════════════════════════════════════════════")
    print(f"  Backend API  →  http://{FASTAPI_HOST}:{FASTAPI_PORT}")
    print(f"  Docs (Swagger) →  http://{FASTAPI_HOST}:{FASTAPI_PORT}/docs")
    print(f"  Streamlit UI →  http://localhost:{STREAMLIT_PORT}")
    print("══════════════════════════════════════════════")
    print("  Press Ctrl+C to stop both services.")
    print()

    # Forward termination signals to child processes
    def shutdown(signum, frame):
        print("\n🛑  Shutting down services …")
        api_proc.terminate()
        ui_proc.terminate()
        api_proc.wait()
        ui_proc.wait()
        print("✅  All services stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep alive and surface any crashes
    while True:
        if api_proc.poll() is not None:
            print("❌  FastAPI backend crashed unexpectedly.")
            ui_proc.terminate()
            sys.exit(1)
        if ui_proc.poll() is not None:
            print("❌  Streamlit UI crashed unexpectedly.")
            api_proc.terminate()
            sys.exit(1)
        time.sleep(2)


if __name__ == "__main__":
    main()
