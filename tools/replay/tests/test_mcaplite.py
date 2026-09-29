"""mcaplite against the official MCAP writer (skipped where the mcap package is not installed, e.g. under Bazel)."""
import io
import json

import pytest

from tools.replay import mcaplite


def write(tmp_path, compression):
    mcap = pytest.importorskip("mcap.writer")
    buf = io.BytesIO()
    w = mcap.Writer(buf, compression=compression, chunk_size=200)
    w.start()
    sid = w.register_schema(name="s", encoding="jsonschema", data=b"{}")
    a = w.register_channel(topic="/a", message_encoding="json", schema_id=sid)
    b = w.register_channel(topic="/b", message_encoding="json", schema_id=sid)
    # written out of time order across channels on purpose
    for i, (ch, t) in enumerate([(a, 30), (b, 10), (a, 20), (b, 40)] * 5):
        w.add_message(ch, log_time=t * 1000 + i, publish_time=0, sequence=i, data=json.dumps({"i": i}).encode())
    w.finish()
    p = tmp_path / "t.mcap"
    p.write_bytes(buf.getvalue())
    return p


def test_matches_official_reader(tmp_path):
    pytest.importorskip("mcap")
    from mcap.reader import make_reader
    from mcap.writer import CompressionType

    p = write(tmp_path, CompressionType.NONE)
    with open(p, "rb") as f:
        want = sorted((c.topic, m.log_time, m.data) for _, c, m in make_reader(f).iter_messages())
    got = list(mcaplite.iter_messages(p))
    assert sorted(got) == want
    assert [t for _, t, _ in got] == sorted(t for _, t, _ in got)
    assert all(tp == "/a" for tp, _, _ in mcaplite.iter_messages(p, {"/a"}))


def test_rejects_compressed_chunks(tmp_path):
    pytest.importorskip("zstandard")
    from mcap.writer import CompressionType

    p = write(tmp_path, CompressionType.ZSTD)
    with pytest.raises(ValueError, match="compressed"):
        list(mcaplite.iter_messages(p))


def test_rejects_non_mcap(tmp_path):
    p = tmp_path / "x.mcap"
    p.write_bytes(b"not an mcap file at all")
    with pytest.raises(ValueError):
        list(mcaplite.iter_messages(p))
