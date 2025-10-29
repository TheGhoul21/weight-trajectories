# wt python-path

Overview
- Prints the Python interpreter path used by `wt.sh` (handy for troubleshooting environments).

How to Run
- `./wt.sh python-path`

Usage
- `./wt.sh python-path`

Resolution order
1) If env var `PYTHON_BIN` is set, that path is used
2) `.venv/bin/python3` under repo root
3) `.venv/bin/python`
4) `python3` on PATH
5) `python` on PATH

Notes
- Use `PYTHON_BIN=/path/to/python ./wt.sh <command>` to force a different interpreter at runtime.
