"""VS Code friendly entry point. Uses the selected interpreter and local source."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from taklatoon.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("play")
    main()

