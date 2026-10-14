"""Safe upload persistence: client file names are ignored, size is enforced while streaming, type is sniffed."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from deepverify.utils.errors import InvalidMedia, PayloadTooLarge, UnsupportedMedia


async def save_upload(upload, dest_dir: Path, max_bytes: int, sniff: Callable[[bytes], str | None], label: str) -> tuple:
    """Stream `upload` (anything with `async read(n)`) into dest_dir/upload.bin. Returns (path, detected_format)."""
    dest, total, fmt = Path(dest_dir) / "upload.bin", 0, None   # fixed name: nothing from the client reaches the filesystem
    with open(dest, "wb") as out:
        while True:
            chunk = await upload.read(1 << 20)
            if not chunk:
                break
            if total == 0:
                fmt = sniff(chunk[:32])
                if fmt is None:
                    raise UnsupportedMedia(f"Unsupported or unrecognised {label} format (content check failed).")
            total += len(chunk)
            if total > max_bytes:
                raise PayloadTooLarge(f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit for {label}s.")
            out.write(chunk)
    if total == 0:
        raise InvalidMedia("The uploaded file is empty.")
    return dest, fmt
