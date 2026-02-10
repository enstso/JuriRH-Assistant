from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .utils import iter_text_files


def load_corpus(input_dir: str | Path) -> List[Dict]:
    """Charge un corpus simple à partir de .txt/.md.
    Retourne une liste de docs: {doc_id, path, text, metadata}
    """
    docs: List[Dict] = []
    for p in iter_text_files(input_dir):
        text = p.read_text(encoding="utf-8", errors="ignore")
        # métadonnées basiques à partir du chemin
        parts = [x for x in p.parts]
        country = None
        if "FR" in parts:
            country = "FR"
        metadata = {
            "country": country or "UNKNOWN",
            "source_path": str(p),
            "filename": p.stem,
        }
        docs.append({
            "doc_id": p.stem,
            "path": str(p),
            "text": text,
            "metadata": metadata,
        })
    return docs
