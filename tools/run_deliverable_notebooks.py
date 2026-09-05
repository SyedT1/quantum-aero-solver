"""Execute the six deliverable notebooks in place from clean kernels."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import argparse

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", help="Notebook names; default is every deliverable")
    args = parser.parse_args()
    paths = ([ROOT / "deliverables" / name for name in args.notebooks]
             if args.notebooks else sorted((ROOT / "deliverables").glob("*.ipynb")))
    if not args.notebooks and len(paths) < 6:
        raise RuntimeError(f"expected at least six notebooks, found {len(paths)}")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        start = perf_counter()
        client = NotebookClient(
            notebook,
            timeout=1800,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
        nbformat.write(notebook, path)
        elapsed = perf_counter() - start
        print(f"PASS {path.relative_to(ROOT)} {elapsed:.2f}s")


if __name__ == "__main__":
    main()
