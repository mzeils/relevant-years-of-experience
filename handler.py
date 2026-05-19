"""RunPod serverless handler for TechWolf/JobBERT-v2.

Supports two tasks:
- embed: encode a list of strings into vectors
- match: rank candidates against a query by cosine similarity
"""

import os
from typing import Any

import runpod
import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = os.environ.get("MODEL_NAME", "TechWolf/JobBERT-v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {DEVICE}...", flush=True)
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
_get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
EMBED_DIM = _get_dim()
print(f"Model loaded. Embedding dim: {EMBED_DIM}", flush=True)


def _embed(texts: list[str], normalize: bool = True) -> list[list[float]]:
    if not texts:
        return []
    vectors = model.encode(
        texts,
        convert_to_tensor=False,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def _match(query: str, candidates: list[str], top_k: int | None) -> list[dict[str, Any]]:
    if not candidates:
        return []
    query_emb = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    cand_emb = model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, cand_emb)[0]
    k = min(top_k or len(candidates), len(candidates))
    top = torch.topk(scores, k=k)
    return [
        {"candidate": candidates[int(idx)], "score": float(score), "index": int(idx)}
        for score, idx in zip(top.values, top.indices)
    ]


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input") or {}
    task = (payload.get("task") or "embed").lower()

    if task == "embed":
        texts = payload.get("texts") or []
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return {"error": "`texts` must be a list of strings"}
        normalize = bool(payload.get("normalize", True))
        return {
            "task": "embed",
            "model": MODEL_NAME,
            "dim": EMBED_DIM,
            "count": len(texts),
            "embeddings": _embed(texts, normalize=normalize),
        }

    if task == "match":
        query = payload.get("query")
        candidates = payload.get("candidates") or []
        top_k = payload.get("top_k")
        if not isinstance(query, str) or not query:
            return {"error": "`query` must be a non-empty string"}
        if not isinstance(candidates, list) or not all(isinstance(c, str) for c in candidates):
            return {"error": "`candidates` must be a list of strings"}
        return {
            "task": "match",
            "model": MODEL_NAME,
            "query": query,
            "results": _match(query, candidates, top_k),
        }

    return {"error": f"unknown task '{task}'. Use 'embed' or 'match'."}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
