"""Minimal MCAP reader for the tooling side (standard library only).

Reads the records drive-replay recordings use: Channel (0x04), Message (0x05) and uncompressed Chunks (0x06).
Anything else is skipped by its length prefix, as the MCAP spec requires. The C++ stack uses the official
reader; tests/test_mcaplite.py checks this one against the official Python package.
"""
from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

MAGIC = b"\x89MCAP0\r\n"
OP_CHANNEL, OP_MESSAGE, OP_CHUNK = 0x04, 0x05, 0x06


def _str(buf: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", buf, off)
    return buf[off + 4: off + 4 + n].decode(), off + 4 + n


def _records(buf: bytes, off: int, end: int) -> Iterator[tuple[int, bytes]]:
    while off + 9 <= end:
        op, n = struct.unpack_from("<BQ", buf, off)
        yield op, buf[off + 9: off + 9 + n]
        off += 9 + n


def iter_messages(path: str | Path, topics: set[str] | None = None) -> Iterator[tuple[str, int, bytes]]:
    """Yields (topic, log_time_ns, payload) in log-time order."""
    buf = Path(path).read_bytes()
    if buf[:8] != MAGIC or buf[-8:] != MAGIC:
        raise ValueError(f"{path}: not an MCAP file")
    channels: dict[int, str] = {}
    out: list[tuple[int, int, str, bytes]] = []

    def visit(records: Iterator[tuple[int, bytes]]) -> None:
        for op, body in records:
            if op == OP_CHANNEL:
                (cid,) = struct.unpack_from("<H", body, 0)
                topic, _ = _str(body, 4)
                channels[cid] = topic
            elif op == OP_MESSAGE:
                cid, seq, log_time = struct.unpack_from("<HIQ", body, 0)
                topic = channels[cid]
                if topics is None or topic in topics:
                    out.append((log_time, seq, topic, body[22:]))
            elif op == OP_CHUNK:
                off = 8 + 8 + 8 + 4
                compression, off = _str(body, off)
                if compression:
                    raise ValueError(f"{path}: compressed chunks ({compression}) are not supported")
                (n,) = struct.unpack_from("<Q", body, off)
                visit(_records(body, off + 8, off + 8 + n))

    visit(_records(buf, 8, len(buf) - 8))
    out.sort(key=lambda r: (r[0], r[1]))
    for log_time, _, topic, data in out:
        yield topic, log_time, data
