from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.config import load_config
from src.retrieval import HybridRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    emb = cfg.get("embeddings", {})
    retriever = HybridRetriever(index_dir=args.index_dir, emb_model_name=emb.get("model_name", "intfloat/multilingual-e5-small"))

    total = 0
    hit = 0
    rows = []
    with Path(args.dataset).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            q = ex["question"]
            filters = ex.get("filters", {})
            expected_hint = ex.get("expected_doc_hint", "")

            chunks = retriever.search(q, filters, top_k_dense=8, top_k_bm25=12, top_k_final=max(args.k, 8), alpha=0.55)
            top_docs = [c.doc_id for c in chunks[:args.k]]

            ok = any(expected_hint in d for d in top_docs)
            total += 1
            hit += int(ok)
            rows.append({"id": ex.get("id"), "ok": ok, "top_docs": top_docs, "expected": expected_hint})

    recall = hit / total if total else 0.0
    print(f"Recall@{args.k}: {recall:.3f} ({hit}/{total})")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
