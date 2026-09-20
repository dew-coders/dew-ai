"""Long-term dataset registry — Dew AI never forgets its training data.

Every dataset the model is trained on is registered here. The registry is
"long-term memory" for the training pipeline:

- Datasets are downloaded once into `data/datasets/` (raw JSONL + plain-text
  corpus + a manifest with row counts, sizes and content hashes).
- On every boot `ensure_all()` verifies the local copies against the manifest
  and automatically re-downloads anything missing or corrupted.
- `train.py` always rebuilds the training corpus FROM THE REGISTRY, so a
  fresh re-train (e.g. after weights were deleted) automatically includes
  every registered dataset — nothing is ever lost.

Public-API datasets are pulled through the Hugging Face datasets-server
(no auth needed, 100 rows per page, politely rate-limited).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from app import config, languages

DATA_DIR = config.DATA_DIR / "datasets"
MANIFEST_PATH = DATA_DIR / "manifest.json"

API = "https://datasets-server.huggingface.co/rows"
PAGE = 100

# --------------------------------------------------------------------------- #
# The long-term training memory. Add new datasets here and everything else
# (download, verification, corpus build, training) picks them up automatically.
# --------------------------------------------------------------------------- #
REGISTRY: List[Dict] = [
    {
        "name": "sinhala-asr-1000",
        "hf_id": "Ransaka/SinhalaASR-1000",
        "text_field": "sentence",
        "max_rows": 1000,
        "kind": "asr",          # ASR transcriptions → plain LM text
        "enabled": True,
    },
    {
        "name": "sinhala-news-24k",
        "hf_id": "NLPC-UOM/Sinhala-News-Source-classification",
        "text_field": "comment",
        "label_field": "label",
        "max_rows": 1000,       # raises the rows fetched per training cycle
        "kind": "news",         # news headlines → knowledge snippets
        "enabled": True,
    },
]


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest() -> Dict:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"datasets": {}}


def save_manifest(manifest: Dict) -> None:
    _ensure_dir()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_ok(entry: Dict, rows: int, raw_path: Path, corpus_path: Path) -> None:
    manifest = load_manifest()
    manifest["datasets"][entry["name"]] = {
        "hf_id": entry["hf_id"],
        "kind": entry["kind"],
        "rows": rows,
        "text_field": entry["text_field"],
        "raw_file": raw_path.name,
        "corpus_file": corpus_path.name,
        "raw_sha256": _hash_file(raw_path),
        "corpus_sha256": _hash_file(corpus_path),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_verified": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_manifest(manifest)


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
def fetch_rows(entry: Dict) -> List[Dict]:
    """Download rows from the HF datasets-server (no auth, paginated)."""
    rows: List[Dict] = []
    offset = 0
    while offset < entry.get("max_rows", 1000):
        length = min(PAGE, entry["max_rows"] - offset)
        for attempt in range(3):
            try:
                resp = requests.get(API, params={
                    "dataset": entry["hf_id"], "config": "default",
                    "split": "train", "offset": offset, "length": length,
                }, timeout=20)
                resp.raise_for_status()
                batch = resp.json().get("rows", [])
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        time.sleep(0.2)
    return [r["row"] for r in rows]


def corpus_text_for(entry: Dict, rows: List[Dict]) -> str:
    """Turn raw rows into training text (kind-specific)."""
    field = entry["text_field"]
    texts = [str(r.get(field) or "").strip() for r in rows]
    texts = [t for t in texts if t]
    if entry["kind"] == "news":
        # headlines stay short — one per line is ideal LM data
        return "\n".join(texts) + "\n"
    if entry["kind"] == "asr":
        # keep the conversational User/Bot pairs so the model learns the
        # chat format in Sinhala, not just raw transcriptions
        try:
            from app.import_dataset import _SI_PROMPTS, _SI_EXTRA
            parts = []
            for kind, prompt in _SI_PROMPTS.items():
                for reply in languages.TEMPLATES["si"].get(kind, []):
                    parts.append(f"User: {prompt}\nBot: {reply}\n")
            for prompt, reply in _SI_EXTRA:
                parts.append(f"User: {prompt}\nBot: {reply}\n")
            return "\n".join(parts) + "\n" + "\n".join(texts) + "\n"
        except Exception:
            pass
    return "\n".join(texts) + "\n"


def download(entry: Dict) -> Dict:
    """Download a dataset, write raw JSONL + corpus text, update the manifest."""
    _ensure_dir()
    raw_path = DATA_DIR / f"{entry['name']}.jsonl"
    corpus_path = DATA_DIR / f"{entry['name']}.txt"
    rows = fetch_rows(entry)
    raw_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    corpus_path.write_text(corpus_text_for(entry, rows), encoding="utf-8")
    _record_ok(entry, len(rows), raw_path, corpus_path)
    return {"name": entry["name"], "rows": len(rows),
            "corpus": str(corpus_path), "raw": str(raw_path)}


# --------------------------------------------------------------------------- #
# Verification / auto-heal
# --------------------------------------------------------------------------- #
def verify(entry: Dict) -> Optional[str]:
    """Return None if the dataset is intact, else a problem description."""
    manifest = load_manifest().get("datasets", {}).get(entry["name"])
    raw_path = DATA_DIR / f"{entry['name']}.jsonl"
    corpus_path = DATA_DIR / f"{entry['name']}.txt"
    if manifest is None:
        return "not in manifest"
    if not raw_path.exists() or not corpus_path.exists():
        return "files missing"
    if _hash_file(corpus_path) != manifest.get("corpus_sha256"):
        return "corpus hash mismatch"
    return None


def ensure_one(entry: Dict) -> Dict:
    """Make sure one dataset is present and intact; re-download if not."""
    name = entry["name"]
    problem = verify(entry)
    if problem:
        print(f"[datasets] {name}: {problem} — re-downloading …", flush=True)
        try:
            info = download(entry)
            info["status"] = "re-downloaded"
            return info
        except Exception as exc:
            return {"name": name, "status": "error", "error": str(exc)[:200]}
    return {"name": name, "status": "ok",
            "rows": load_manifest()["datasets"][name]["rows"]}


def ensure_all() -> List[Dict]:
    """Verify + auto-heal every enabled registry dataset (called on boot)."""
    return [ensure_one(entry) for entry in REGISTRY if entry.get("enabled", True)]


# --------------------------------------------------------------------------- #
# Corpus access
# --------------------------------------------------------------------------- #
def corpus_paths() -> List[Path]:
    """All enabled dataset corpus files (consumed by train.py and knowledge.py)."""
    paths = []
    for entry in REGISTRY:
        if not entry.get("enabled", True):
            continue
        p = DATA_DIR / f"{entry['name']}.txt"
        if p.exists():
            paths.append(p)
    return paths


def load_rows(name: str) -> List[Dict]:
    """Raw stored rows for a dataset (from the local JSONL copy)."""
    raw_path = DATA_DIR / f"{name}.jsonl"
    if not raw_path.exists():
        return []
    rows = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def status() -> List[Dict]:
    manifest = load_manifest().get("datasets", {})
    out = []
    for entry in REGISTRY:
        m = manifest.get(entry["name"], {})
        problem = verify(entry)
        out.append({
            "name": entry["name"],
            "hf_id": entry["hf_id"],
            "kind": entry["kind"],
            "enabled": entry.get("enabled", True),
            "rows": m.get("rows", 0),
            "updated_at": m.get("updated_at"),
            "healthy": problem is None,
            "issue": problem,
        })
    return out


# --------------------------------------------------------------------------- #
# CLI:  python -m app.datasets  [name]
# --------------------------------------------------------------------------- #
def _cli() -> None:
    import sys
    targets = REGISTRY if len(sys.argv) < 2 else [
        e for e in REGISTRY if e["name"] == sys.argv[1]]
    if not targets:
        print("unknown dataset:", sys.argv[1])
        return
    for entry in targets:
        info = ensure_one(entry)
        print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
