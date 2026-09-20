"""Image generation for Dew AI.

Two providers, chosen automatically:
1. Pollinations.ai (free, keyless) — photorealistic AI images via a simple
   GET URL. No API key required, safe for public demos.
2. Offline SVG art fallback — a deterministic generative "art" renderer that
   always works (no network): gradients, shapes and palette derived from the
   prompt hash. Used when the API is unreachable.

Images are cached on disk (data/images/) and referenced by URL.
"""
from __future__ import annotations

import hashlib
import re
import time
import zlib
from pathlib import Path
from typing import Dict, Optional

import requests

from app import config

IMG_DIR = config.DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&nologo=true&seed={seed}"

# simple prompt-injection guard: strip anything that looks like instructions
_INJECT_RE = re.compile(r"(ignore|forget)\s+(all\s+)?(previous|prior|above)?\s*(instructions|prompts?)", re.I)


def _safe_name(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:60].strip("-") or "image"
    return f"{slug}-{int(time.time())}.png"


def _palette(seed_text: str):
    h = zlib.crc32(seed_text.encode("utf-8"))
    hues = [(h >> i) % 360 for i in (0, 8, 16)]
    return [f"hsl({hue}, 70%, {55 + (h >> (i + 4)) % 20}%)" for i, hue in enumerate(hues)]


def _svg_art(prompt: str) -> str:
    """Deterministic generative art (offline fallback)."""
    p1, p2, p3 = _palette(prompt)
    shapes = []
    h = zlib.crc32(prompt.encode("utf-8"))
    for i in range(14):
        cx, cy = (h >> (i * 3)) % 800, (h >> (i * 3 + 5)) % 600
        r = 40 + (h >> (i * 2)) % 160
        color = [p1, p2, p3][(h >> i) % 3]
        shapes.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" opacity="0.35"/>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="{p1}"/><stop offset="100%" stop-color="{p2}"/>
</linearGradient></defs>
<rect width="800" height="600" fill="url(#bg)"/>
{''.join(shapes)}
<text x="40" y="560" font-family="Segoe UI, sans-serif" font-size="22" fill="white" opacity="0.9">{prompt[:60]}</text>
</svg>"""


def generate(prompt: str, width: int = 768, height: int = 512) -> Dict:
    """Generate an image for a prompt. Returns {url, provider, cached}."""
    prompt = _INJECT_RE.sub("", prompt).strip()[:300]
    if not prompt:
        prompt = "abstract art"

    seed = zlib.crc32(prompt.encode("utf-8")) % 100000
    fname = _safe_name(prompt)
    path = IMG_DIR / fname

    # 1) try the free AI image API
    try:
        resp = requests.get(
            POLLINATIONS_URL.format(prompt=requests.utils.quote(prompt),
                                    w=width, h=height, seed=seed),
            timeout=25)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
            path.write_bytes(resp.content)
            return {"url": f"/images/{fname}", "provider": "pollinations",
                    "cached": False}
    except requests.RequestException:
        pass

    # 2) offline deterministic SVG art
    svg = _svg_art(prompt)
    fname = fname.replace(".png", ".svg")
    (IMG_DIR / fname).write_text(svg, encoding="utf-8")
    return {"url": f"/images/{fname}", "provider": "offline-art", "cached": False}


def serve_path(fname: str) -> Optional[Path]:
    """Resolve a cached image filename (safety-checked)."""
    path = (IMG_DIR / fname).resolve()
    if path.parent != IMG_DIR.resolve() or not path.exists():
        return None
    return path
