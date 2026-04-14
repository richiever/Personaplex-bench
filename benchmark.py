"""Thin shim: `python benchmark.py ...` -> bench.cli.main."""

from bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
