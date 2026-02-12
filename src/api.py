from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import load_config
from .llm_client import make_llm
from .rag import ask_rag
from .retrieval import HybridRetriever

app = FastAPI(title="Assistant juridique RH (RAG) — OpenSource")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    filters: Dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def get_cfg() -> dict:
    return load_config("config.yaml")  # <-- dict

@lru_cache(maxsize=1)
def get_llm():
    cfg = get_cfg()
    return make_llm(cfg)

@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    cfg = get_cfg()
    idx_dir = cfg.get("paths", {}).get("index_dir", "data/index")
    emb_name = cfg.get("embeddings", {}).get("model_name", "intfloat/multilingual-e5-small")
    return HybridRetriever(idx_dir, emb_name)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    cfg = get_cfg()
    llm = get_llm()
    retriever = get_retriever()

    retr_cfg = cfg.get("retrieval", {})
    sec = cfg.get("security", {})
    llm_cfg = cfg.get("llm", {})

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
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=int(llm_cfg.get("max_tokens", 700)),
        refuse_if_no_context=bool(sec.get("refuse_if_no_context", True)),
    )
    return out
