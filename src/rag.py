from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .llm_client import BaseLLMClient
from .retrieval import RetrievedChunk


SYSTEM_PROMPT = (
    "Tu es un assistant juridique RH. "
    "Tu dois répondre uniquement à partir des extraits de documents fournis. "
    "Si l'information n'est pas présente dans les extraits, dis clairement que tu ne peux pas répondre avec certitude "
    "et propose des pistes (quel document/quelle info demander). "
    "Tu cites tes sources sous la forme [doc_id]. "
    "Tu ne donnes pas de conseil juridique définitif; tu proposes une formulation prudente."
)


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.doc_id}] {c.text}")
    return "\n\n".join(parts)


def ask_rag(llm: BaseLLMClient, question: str, chunks: List[RetrievedChunk],
            temperature: float, max_tokens: int, refuse_if_no_context: bool = True) -> Dict[str, Any]:
    if refuse_if_no_context and not chunks:
        return {
            "answer": "Je ne peux pas répondre de façon fiable car je n'ai trouvé aucun extrait pertinent dans la base documentaire.",
            "citations": [],
            "used_chunks": [],
        }

    context = build_context(chunks)
    user_prompt = (
        "Extraits (base documentaire):\n"
        f"{context}\n\n"
        "Question utilisateur:\n"
        f"{question}\n\n"
        "Règles de réponse:\n"
        "- Réponds en français, de manière structurée (2-6 puces max + une phrase de prudence).\n"
        "- Chaque affirmation importante doit être soutenue par une citation [doc_id].\n"
        "- Si une info manque, dis-le clairement.\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    answer = llm.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)

    # extraction citations naïve: [doc_id]
    citations = []
    for c in chunks:
        if f"[{c.doc_id}]" in answer and c.doc_id not in citations:
            citations.append(c.doc_id)

    return {
        "answer": answer.strip(),
        "citations": citations,
        "used_chunks": [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": c.score, "metadata": c.metadata}
            for c in chunks
        ],
    }
