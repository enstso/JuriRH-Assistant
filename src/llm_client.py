from __future__ import annotations

from typing import Dict, List

import httpx


class LLMError(RuntimeError):
    pass


class BaseLLMClient:
    def chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    """Client pour Ollama (API locale) : POST /api/chat"""
    def __init__(self, base_url: str, model: str, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, json=payload)
            if r.status_code >= 400:
                raise LLMError(f"Ollama error {r.status_code}: {r.text[:500]}")
            data = r.json()
        try:
            return data["message"]["content"]
        except Exception as e:
            raise LLMError(f"Réponse Ollama inattendue: {data}") from e


class VllmOpenAIClient(BaseLLMClient):
    """Client pour serveur vLLM exposant une API compatible OpenAI."""
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise LLMError(f"vLLM error {r.status_code}: {r.text[:500]}")
            data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"Réponse vLLM inattendue: {data}") from e


def make_llm(cfg: Dict) -> BaseLLMClient:
    llm_cfg = cfg.get("llm", {})
    backend = llm_cfg.get("backend", "ollama")
    model = llm_cfg.get("model", "mistral")

    if backend == "ollama":
        return OllamaClient(
            base_url=llm_cfg.get("ollama_base_url", "http://localhost:11434"),
            model=model,
        )
    if backend == "vllm_openai":
        return VllmOpenAIClient(
            base_url=llm_cfg["base_url"],
            model=model,
            api_key=llm_cfg.get("api_key", ""),
        )

    raise ValueError(f"Backend LLM inconnu: {backend}")
