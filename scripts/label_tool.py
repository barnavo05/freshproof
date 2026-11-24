"""Open the labeling tool in your browser.

    pip install -r requirements-api.txt
    PYTHONPATH=src python scripts/label_tool.py --data-root data

Put your photos in <data-root>/raw first. Labels go to <data-root>/labels.csv
and defect masks to <data-root>/masks.
"""

import argparse
import threading
import webbrowser
from pathlib import Path

import uvicorn

from freshproof.labeling.app import create_labeling_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Labeling tool running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    # Bound to 127.0.0.1 so only your own computer can reach it.
    uvicorn.run(create_labeling_app(root), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
