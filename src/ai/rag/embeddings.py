"""Local embedding model wrapper (sentence-transformers).

Loaded once per process (module-level lazy singleton, mirroring
src/ai/tools/_artifacts.py's lru_cache pattern for the churn/recommender
artifacts) - the model download/load is what makes this dependency
heavy, so it must never happen more than once per process and must never
happen at import time (every test and every module that imports this one
must not be forced to load a ~90MB model just to be imported).

Kept out of docker/requirements-airflow.txt on purpose: only the API
service ever embeds a live query at request time (see docker-compose.yml's
"serving" profile).
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

EMBEDDING_MODEL_NAME = os.environ.get("AI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2's output dimension


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (n, EMBEDDING_DIM) float32 array, L2-normalized so cosine
    similarity and inner product rank identically - pgvector's <=>
    (cosine distance) operator works directly against these with no
    separate normalization step at query time."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
