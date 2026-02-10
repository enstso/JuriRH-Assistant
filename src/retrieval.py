from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .utils import normalize_whitespace


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class HybridRetriever:
    def __init__(self, index_dir: str | Path, emb_model_name: str):
        index_dir = Path(index_dir)

        self.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        self.model = SentenceTransformer(emb_model_name)

        # chunks
        chunks = []
        with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        self.chunks = chunks

        # bm25 tokens
        tokenized = []
        with (index_dir / "bm25_tokens.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tokenized.append(json.loads(line))
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, filters: Dict[str, Any] | None, *,
               top_k_dense: int, top_k_bm25: int, top_k_final: int, alpha: float) -> List[RetrievedChunk]:
        filters = filters or {}

        # Pre-filter candidates by metadata (country, etc.)
        allowed_idx = []
        for i, c in enumerate(self.chunks):
            ok = True
            for k, v in filters.items():
                if v is None:
                    continue
                if c.get("metadata", {}).get(k) != v:
                    ok = False
                    break
            if ok:
                allowed_idx.append(i)

        if not allowed_idx:
            return []

        # BM25 scores for allowed
        q_tokens = normalize_whitespace(query).lower().split()
        bm25_scores_all = self.bm25.get_scores(q_tokens)
        bm25_scores = [(i, float(bm25_scores_all[i])) for i in allowed_idx]
        bm25_scores.sort(key=lambda x: x[1], reverse=True)
        bm25_top = bm25_scores[:top_k_bm25]

        # Dense scores (FAISS) — we query full index, then filter down to allowed
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        D, I = self.faiss_index.search(q_emb, k=min(200, len(self.chunks)))
        dense = []
        allowed_set = set(allowed_idx)
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx in allowed_set:
                dense.append((idx, float(score)))
            if len(dense) >= top_k_dense:
                break

        # Normalize & fuse
        def minmax(norm_list: List[Tuple[int, float]]) -> Dict[int, float]:
            if not norm_list:
                return {}
            vals = [s for _, s in norm_list]
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                return {i: 1.0 for i, _ in norm_list}
            return {i: (s - lo) / (hi - lo) for i, s in norm_list}

        bm25_n = minmax(bm25_top)
        dense_n = minmax(dense)

        fused: Dict[int, float] = {}
        for idx, s in bm25_n.items():
            fused[idx] = fused.get(idx, 0.0) + (1 - alpha) * s
        for idx, s in dense_n.items():
            fused[idx] = fused.get(idx, 0.0) + alpha * s

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k_final]
        out: List[RetrievedChunk] = []
        for idx, score in ranked:
            c = self.chunks[idx]
            out.append(RetrievedChunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                text=c["text"],
                metadata=c.get("metadata", {}),
                score=float(score),
            ))
        return out
