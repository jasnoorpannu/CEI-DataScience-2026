from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "app.py"


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8501",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
