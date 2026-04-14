#!/usr/bin/env python3
"""Standalone web server launcher for the CAR pipeline.

This script adds the project root to sys.path so it can be run directly
without pip-installing the package. Usage:
    python run_web.py [--host HOST] [--port PORT]
"""

import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Install torch mock BEFORE any pgmpy imports (pgmpy requires torch at import
# time but we only need the numpy backend).
from car.torch_mock import install as _install_torch_mock
_install_torch_mock()

# Set matplotlib backend before any imports that might touch matplotlib
import matplotlib
matplotlib.use("Agg")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    host = os.environ.get("CAR_HOST", "127.0.0.1")
    port = int(os.environ.get("CAR_PORT", "5000"))

    # Parse simple CLI args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--host",) and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] in ("--port", "-p") and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        else:
            i += 1

    from car.web.app import app

    print(f"Starting CAR web interface at http://{host}:{port}")
    print(f"Open your browser and navigate to the URL above.")
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()
