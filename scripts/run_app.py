"""Launch the Bird Aero Vibration Analysis demo."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    """Run Streamlit app via subprocess."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/vibration_poc/app.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
