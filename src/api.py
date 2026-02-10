from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from .config import load_config
from .llm_client import make_llm
from .retrieval import HybridRetriever
from .rag import ask_rag


app = FastAPI(title="Assistant juridique RH (RAG) — OpenSource")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    filters: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    cfg = load_config("config.yaml")
    llm = make_llm(cfg)

    paths = cfg.get("paths", {})
    idx_dir = paths.get("index_dir", "data/index")

    emb = cfg.get("embeddings", {})
    retr_cfg = cfg.get("retrieval", {})
    sec = cfg.get("security", {})

    retriever = HybridRetriever(index_dir=idx_dir, emb_model_name=emb.get("model_name", "intfloat/multilingual-e5-small"))

    chunks = retriever.search(
        query=req.question,
        filters=req.filters,
        top_k_dense=int(retr_cfg.get("top_k_dense", 8)),
        top_k_bm25=int(retr_cfg.get("top_k_bm25", 12)),
        top_k_final=int(retr_cfg.get("top_k_final", 8)),
        alpha=float(retr_cfg.get("alpha", 0.55)),
    )

    out = ask_rag(
        llm=llm,
        question=req.question,
        chunks=chunks,
        temperature=float(cfg.get("llm", {}).get("temperature", 0.2)),
        max_tokens=int(cfg.get("llm", {}).get("max_tokens", 700)),
        refuse_if_no_context=bool(sec.get("refuse_if_no_context", True)),
    )

    return out
