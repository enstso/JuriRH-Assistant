from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

from .utils import normalize_whitespace, stable_id


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """Chunking simple en caractères (robuste sur docs texte).
    On coupe sur des limites "propres" quand possible.
    """
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        # tente de couper sur un séparateur
        cut = j
        for sep in ["\n\n", "\n", ". ", "; "]:
            k = text.rfind(sep, i, j)
            if k != -1 and k > i + 200:
                cut = k + len(sep)
                break
        chunk = text[i:cut].strip()
        if chunk:
            chunks.append(chunk)
        i = max(cut - overlap, i + 1)
    return chunks


def build_indexes(docs: List[Dict[str, Any]], out_dir: str | Path, emb_model_name: str, batch_size: int,
                  chunk_size: int, overlap: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Chunking
    chunks: List[Chunk] = []
    for d in docs:
        doc_id = d["doc_id"]
        meta = d.get("metadata", {})
        for idx, c in enumerate(chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)):
            chunk_id = stable_id(f"{doc_id}:{idx}:{c[:80]}")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=c,
                metadata=meta | {"chunk_index": idx}
            ))

    # 2) BM25
    tokenized = [normalize_whitespace(c.text).lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # 3) Embeddings + FAISS
    model = SentenceTransformer(emb_model_name)
    texts = [c.text for c in chunks]
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # Persist
    faiss.write_index(index, str(out_dir / "faiss.index"))
    # BM25 est sérialisé en sauvegardant tokens + idf/avgdl via pickle, mais on évite pickle:
    # on reconstruit à partir des tokens.
    (out_dir / "bm25_tokens.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tokenized) + "\n",
        encoding="utf-8"
    )
    # Chunks metadata
    chunk_records = [{
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "text": c.text,
        "metadata": c.metadata,
    } for c in chunks]
    (out_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in chunk_records) + "\n",
        encoding="utf-8"
    )
