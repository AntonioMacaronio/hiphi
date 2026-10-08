"""Integrity checks for the raw release mirror.

Example, from hiphi:
    source scripts/project_env.sh
    PYTHONPATH=scripts "$HIPHI_PYTHON" -m pytest -q tests/test_mirror_raw_release.py
"""

import base64
import hashlib
from types import SimpleNamespace

import mirror_raw_release as mirror
import pytest


def entry(body, lfs=True):
    return dict(
        path="data/example",
        bytes=len(body),
        lfs_sha256=hashlib.sha256(body).hexdigest() if lfs else None,
        git_blob_sha1=hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest(),
    )


@pytest.mark.parametrize("lfs", [False, True])
def test_source_hash_verification_rejects_same_size_corruption(tmp_path, lfs):
    path = tmp_path / "data"
    body = b"original source bytes"
    path.write_bytes(body)
    record = mirror.file_integrity(path, entry(body, lfs))
    assert record["sha256"] == hashlib.sha256(body).hexdigest()
    path.write_bytes(b"X" + body[1:])
    with pytest.raises(ValueError, match="content hash differs"):
        mirror.file_integrity(path, entry(body, lfs))


def test_git_blob_identity_includes_the_git_header(tmp_path):
    path = tmp_path / "data"
    body = b"source"
    path.write_bytes(body)
    expected = entry(body, lfs=False)
    expected["git_blob_sha1"] = hashlib.sha1(body).hexdigest()
    with pytest.raises(ValueError, match="content hash differs"):
        mirror.file_integrity(path, expected)


def test_multipart_checksum_includes_the_final_partial_part(tmp_path, monkeypatch):
    monkeypatch.setattr(mirror, "PART_BYTES", 3)
    monkeypatch.setattr(mirror, "MULTIPART_THRESHOLD", 6)
    path = tmp_path / "data"
    body = b"abcdefgh"
    path.write_bytes(body)
    record = mirror.file_integrity(path, entry(body))
    part_digests = [hashlib.sha256(part).digest() for part in (b"abc", b"def", b"gh")]
    expected = (
        base64.b64encode(hashlib.sha256(b"".join(part_digests)).digest()).decode()
        + "-3"
    )
    assert record["s3_checksum_sha256"] == expected
    assert record["s3_checksum_type"] == "COMPOSITE"


def test_s3_metadata_sha_does_not_replace_server_checksum():
    record = dict(
        path="data/example", bytes=12, sha256="local", s3_checksum_sha256="expected"
    )
    args = SimpleNamespace(prefix="prefix", bucket="bucket")
    client = SimpleNamespace(
        head_object=lambda **kwargs: dict(
            ContentLength=12,
            Metadata={"sha256": "local"},
            ChecksumSHA256="wrong",
        )
    )
    with pytest.raises(ValueError, match="stored SHA256 checksum differs"):
        mirror.verify_remote(client, args, record)


def test_source_path_cannot_escape_raw_root(tmp_path):
    for path in ("../outside", "/absolute", "data/../../outside"):
        with pytest.raises(ValueError):
            mirror.safe_path(tmp_path, path)
