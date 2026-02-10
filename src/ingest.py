from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .loaders import load_corpus
from .indexing import build_indexes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    docs = load_corpus(args.input_dir)

    emb = cfg.get("embeddings", {})
    ch = cfg.get("chunking", {})

    build_indexes(
        docs=docs,
        out_dir=args.out_dir,
        emb_model_name=emb.get("model_name", "intfloat/multilingual-e5-small"),
        batch_size=int(emb.get("batch_size", 32)),
        chunk_size=int(ch.get("chunk_size", 900)),
        overlap=int(ch.get("overlap", 120)),
    )

    print(f"✅ Index construit dans: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
