"""Image upload + resolution helpers.

Self-contained media handling: the /upload HTTP endpoint (multipart → file on
disk under uploads/) and turning stored ``/uploads/...`` URLs back into data
URLs for the multimodal LLM call. No runtime backend state is touched here.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import uuid
from pathlib import Path

from aiohttp import web

LOG = logging.getLogger("voice-chat")

HERE = Path(__file__).resolve().parent.parent
UPLOAD_DIR = HERE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_MIME = {
    "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", "image/bmp",
}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024


async def upload_image(request):
    """Accepts an image as multipart/form-data, returns a URL."""
    try:
        reader = await request.multipart()
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"multipart parse: {exc}"}, status=400)

    field_part = await reader.next()
    if field_part is None or field_part.name != "file":
        return web.json_response({"ok": False, "error": "missing field 'file'"}, status=400)

    content_type = (field_part.headers.get("Content-Type") or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_IMAGE_MIME:
        return web.json_response(
            {"ok": False, "error": f"unsupported content type: {content_type or '<none>'}"},
            status=400)

    orig_name = field_part.filename or "upload"
    # NEVER take the suffix from the user filename (XSS protection: evil.html as image/jpeg).
    suffix = {
        "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
        "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp",
    }.get(content_type, ".bin")
    safe_id = uuid.uuid4().hex
    out_path = UPLOAD_DIR / f"{safe_id}{suffix}"

    total = 0
    with out_path.open("wb") as f:
        while True:
            chunk = await field_part.read_chunk(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                f.close()
                try:
                    out_path.unlink()
                except Exception:
                    pass
                return web.json_response(
                    {"ok": False, "error": f"file too large (>{MAX_UPLOAD_BYTES} bytes)"},
                    status=413)
            f.write(chunk)

    rel_url = f"/uploads/{out_path.name}"
    LOG.info("upload accepted: name=%r bytes=%d type=%s -> %s",
             orig_name, total, content_type, rel_url)
    return web.json_response({
        "ok": True, "url": rel_url, "name": orig_name,
        "bytes": total, "contentType": content_type,
    })


def _image_url_to_data_url(url: str) -> str | None:
    if not url:
        return None
    if url.startswith("data:") or url.startswith("http://") or url.startswith("https://"):
        return url
    if not url.startswith("/uploads/"):
        return None
    name = url[len("/uploads/"):]
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        return None
    path = UPLOAD_DIR / name
    if not path.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "application/octet-stream"
    try:
        b = path.read_bytes()
    except Exception:
        return None
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def _resolve_image_urls(image_urls: list, log_tag: str) -> list:
    if not image_urls:
        return []
    results = await asyncio.gather(*(
        asyncio.to_thread(_image_url_to_data_url, u) for u in image_urls
    ))
    out: list = []
    for u, durl in zip(image_urls, results):
        if durl:
            out.append(durl)
        else:
            LOG.warning("%s image %r could not be resolved", log_tag, u)
    return out
