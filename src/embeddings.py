from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_embedder(model_name: str):
    return SentenceTransformer(model_name)
