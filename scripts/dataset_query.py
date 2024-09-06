from __future__ import annotations

import subprocess
import sys


def main():
    # Forward the arguments to the coffea.dataset_tools.dataset_query module
    subprocess.run([sys.executable, "-m", "coffea.dataset_tools.dataset_query"] + sys.argv[1:], check=False)


if __name__ == "__main__":
    main()
