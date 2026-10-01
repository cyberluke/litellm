"""Edge request/response compression (Phase 3 §46 step 12/13).

Producer order: full local request -> Differential Context -> serialize
delta envelope -> zstd/gzip -> HTTP/2. The full logical request is NEVER
compressed and then diffed — the diff happens on logical units first, and
only the delta envelope is compressed.

Response side: SSEProxy may compress non-stream JSON responses
(Content-Encoding); SSE stays identity. Decode both zstd and gzip.
"""

from __future__ import annotations

import gzip
import time
from typing import Optional, Tuple

try:
    import zstandard

    _HAS_ZSTD = True
except ImportError:  # pragma: no cover - environment dependent
    zstandard = None  # type: ignore[assignment]
    _HAS_ZSTD = False

ZSTD_LEVEL = 3
GZIP_LEVEL = 3

SUPPORTED = tuple(e for e in ("zstd", "gzip", "identity") if e != "zstd" or _HAS_ZSTD)


class CompressionError(Exception):
    pass


def compress_body(body: bytes, encoding: str) -> bytes:
    if encoding == "identity":
        return body
    if encoding == "zstd":
        if not _HAS_ZSTD:
            raise CompressionError("zstd unavailable")
        assert zstandard is not None
        return zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(body)
    if encoding == "gzip":
        return gzip.compress(body, compresslevel=GZIP_LEVEL)
    raise CompressionError(f"unsupported compression encoding {encoding!r}")


def decompress_body(body: bytes, content_encoding: Optional[str]) -> bytes:
    """Decode a response body.

    httpx >= 0.28 auto-decompresses zstd/gzip when the codec package is
    installed, and leaves the ``content-encoding`` header in place. Guard
    with magic-byte sniffing so an already-decoded body is returned as-is
    instead of being double-decompressed (Phase 4 smoke: 3.72.19.167).
    """
    encoding = (content_encoding or "").strip().lower()
    if not encoding or encoding == "identity":
        return body
    if encoding == "gzip":
        if not body.startswith(b"\x1f\x8b"):
            # Already decoded by the HTTP client (header retained).
            return body
        import zlib

        try:
            return zlib.decompress(body, 16 + zlib.MAX_WBITS)
        except zlib.error as exc:
            raise CompressionError(f"malformed gzip response: {exc}") from exc
    if encoding == "zstd":
        if not body.startswith(b"\x28\xb5\x2f\xfd"):
            # Already decoded by the HTTP client (header retained).
            return body
        if not _HAS_ZSTD:
            raise CompressionError("zstd unavailable")
        assert zstandard is not None
        try:
            return zstandard.ZstdDecompressor().decompress(body)
        except zstandard.ZstdError as exc:
            raise CompressionError(f"malformed zstd response: {exc}") from exc
    raise CompressionError(f"unsupported response content-encoding {encoding!r}")


def maybe_compress(body: bytes, encoding: str, threshold: int) -> Tuple[bytes, str, float]:
    """Compress only when the body is above the threshold AND compression
    actually shrinks it; otherwise identity. Returns (body, encoding,
    compression_ms)."""
    started = time.monotonic()
    if encoding == "identity" or len(body) < threshold:
        return body, "identity", 0.0
    if encoding not in SUPPORTED:
        return body, "identity", 0.0
    compressed = compress_body(body, encoding)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if len(compressed) >= len(body):
        return body, "identity", elapsed_ms
    return compressed, encoding, elapsed_ms


__all__ = [
    "SUPPORTED",
    "ZSTD_LEVEL",
    "GZIP_LEVEL",
    "CompressionError",
    "compress_body",
    "decompress_body",
    "maybe_compress",
]