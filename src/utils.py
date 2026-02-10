from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def iter_text_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            yield p


def normalize_whitespace(s: str) -> str:
    return " ".join(s.replace("\r", " ").replace("\n", " ").split())
