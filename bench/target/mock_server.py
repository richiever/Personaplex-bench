"""Minimal WS echo server for local smoke tests.

Accepts raw dialect messages and echoes them back unchanged. Enough to
exercise the TargetClient round-trip end-to-end without a real model.

Usage:
    python -m bench.target.mock_server --port 8999
"""

from __future__ import annotations

import argparse
import asyncio


async def _handler(ws):
    async for msg in ws:
        await ws.send(msg)


async def serve(host: str = "127.0.0.1", port: int = 8999) -> None:
    import websockets
    async with websockets.serve(_handler, host, port, max_size=None):
        await asyncio.Future()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8999)
    args = p.parse_args(argv)
    asyncio.run(serve(args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
