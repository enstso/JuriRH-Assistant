from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

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


def _as_dict(x: Any) -> Dict[str, Any]:
    """Convertit chunk/objet en dict si possible, sinon {}."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    # Pydantic / dataclass / objets avec .dict()
    if hasattr(x, "dict") and callable(getattr(x, "dict")):
        try:
            return x.dict()
        except Exception:
            return {}
    # objets simples avec __dict__
    if hasattr(x, "__dict__"):
        try:
            return dict(x.__dict__)
        except Exception:
            return {}
    return {}


def _pick_citation_id(chunk: Any) -> str:
    """
    Essaie de trouver une "citation" stable dans un chunk.
    Priorité: doc_id / source / url / title / id / metadata.*
    """
    d = _as_dict(chunk)
    md = d.get("metadata") or {}
    if not isinstance(md, dict):
        md = {}

    candidates = [
        d.get("doc_id"),
        d.get("docId"),
        d.get("document_id"),
        md.get("doc_id"),
        md.get("docId"),
        md.get("document_id"),
        d.get("source"),
        md.get("source"),
        d.get("url"),
        md.get("url"),
        d.get("title"),
        md.get("title"),
        d.get("id"),
        md.get("id"),
    ]
    for c in candidates:
        if c is None:
            continue
        s = str(c).strip()
        if s and s.lower() != "none":
            return s
    return ""


def _build_citations_from_chunks(chunks: Optional[List[Any]]) -> List[str]:
    """Construit une liste dédupliquée de citations à partir des chunks."""
    if not chunks:
        return []
    seen = set()
    out: List[str] = []
    for ch in chunks:
        c = _pick_citation_id(ch)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


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

    # 1) Retrieval
    chunks = retriever.search(
        query=req.question,
        filters=req.filters,
        top_k_dense=int(retr_cfg.get("top_k_dense", 8)),
        top_k_bm25=int(retr_cfg.get("top_k_bm25", 12)),
        top_k_final=int(retr_cfg.get("top_k_final", 8)),
        alpha=float(retr_cfg.get("alpha", 0.55)),
    )

    # 2) Génération (RAG)
    out = ask_rag(
        llm=llm,
        question=req.question,
        chunks=chunks,
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=int(llm_cfg.get("max_tokens", 700)),
        refuse_if_no_context=bool(sec.get("refuse_if_no_context", True)),
    )

    # 3) Normalisation de la réponse pour l'éval:
    #    - garantir "answer" et "citations"
    #    - garder "context" si présent, sinon injecter les chunks
    if not isinstance(out, dict):
        out = {"answer": str(out)}

    if "answer" not in out:
        out["answer"] = out.get("text") or out.get("response") or ""

    # Context: si ask_rag n'en renvoie pas, on met les chunks (utile debug + eval)
    if "context" not in out or out["context"] is None:
        out["context"] = chunks

    # Citations: si ask_rag n'en renvoie pas, on les construit depuis chunks/context
    if not isinstance(out.get("citations"), list) or len(out.get("citations") or []) == 0:
        out["citations"] = _build_citations_from_chunks(out.get("context"))

    return out
