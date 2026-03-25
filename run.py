"""
Fashion AI Recommender Launcher
Starts backend and opens frontend in browser
"""

import subprocess
import webbrowser
import time
import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    missing = []
    checks = {
        "tensorflow": "tensorflow",
        "sklearn":    "scikit-learn",
        "cv2":        "opencv-python",
        "sentence_transformers": "sentence-transformers",
        "fastapi":    "fastapi",
        "PIL":        "Pillow",
        "numpy":      "numpy",
        "uvicorn":    "uvicorn",
    }
    for module, pkg in checks.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"✗ Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print("  pip install -r requirements_new.txt")
        return False

    print("✓ All dependencies installed")
    return True


def start_backend():
    """Start the FastAPI backend server"""
    print("\n" + "=" * 60)
    print("Starting Fashion AI Backend Server...")
    print("=" * 60)

    backend_path = Path(__file__).parent / "backend.py"
    process = subprocess.Popen(
        [sys.executable, str(backend_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("Waiting for server to initialize (this may take a moment for ML models)...")
    time.sleep(5)

    if process.poll() is None:
        print("✓ Backend server running at http://localhost:8000")
        return process
    else:
        out, err = process.communicate()
        print("✗ Backend failed to start")
        print("STDOUT:", out[:500] if out else "(none)")
        print("STDERR:", err[:500] if err else "(none)")
        return None


def open_frontend():
    """Open frontend in default browser"""
    frontend_path = Path(__file__).parent / "frontend.html"

    if frontend_path.exists():
        print("\n" + "=" * 60)
        print("Opening frontend in browser...")
        print("=" * 60)
        webbrowser.open(f"file://{frontend_path.resolve()}")
        print(f"✓ Frontend opened: {frontend_path}")
    else:
        print("✗ frontend.html not found in", str(frontend_path.parent))


def main():
    print("\n" + "=" * 60)
    print("🎨 AI Fashion Recommender Launcher")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    backend_process = start_backend()

    if backend_process:
        time.sleep(1)
        open_frontend()

        print("\n" + "=" * 60)
        print("System is running!")
        print("=" * 60)
        print("\nBackend:  http://localhost:8000")
        print("API docs: http://localhost:8000/docs")
        print("Frontend: Check your browser")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60)

        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            backend_process.terminate()
            backend_process.wait()
            print("✓ Server stopped")
    else:
        print("\n✗ Failed to start backend")
        sys.exit(1)


if __name__ == "__main__":
    main()
