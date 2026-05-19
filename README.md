# JobBERT-v2 on RunPod Serverless

[![Runpod](https://api.runpod.io/badge/mzeils/relevant-years-of-experience)](https://console.runpod.io/hub/mzeils/relevant-years-of-experience)

Serverless endpoint wrapping [TechWolf/JobBERT-v2](https://huggingface.co/TechWolf/JobBERT-v2) — a sentence-transformers model fine-tuned for job-title matching.

Two tasks:
- **embed** — return vectors for a list of strings
- **match** — rank candidates against a query by cosine similarity

## Layout

```
handler.py         # RunPod serverless entrypoint
Dockerfile         # Bakes the model into the image (no cold-start download)
requirements.txt
test_input.json    # Sample request for `runpod test`
test_local.py      # Local smoke test (no RunPod runtime needed)
```

## Local test (CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_local.py
```

First run downloads the model (~430 MB) to your HF cache.

## Build & push the image

Replace `YOUR_DOCKERHUB_USER` with your registry namespace. RunPod Serverless workers need **linux/amd64**.

```bash
docker buildx build \
  --platform linux/amd64 \
  -t YOUR_DOCKERHUB_USER/jobbert-runpod:latest \
  --push .
```

The build step pre-downloads the model into the image (`HF_HOME=/opt/hf-cache`), so cold starts only pay the model-load cost, not the download.

## Deploy on RunPod

1. Console → **Serverless** → **New Endpoint**.
2. Container image: `YOUR_DOCKERHUB_USER/jobbert-runpod:latest`.
3. GPU: any 16 GB+ card is overkill — JobBERT-v2 is small. **RTX A4000 / RTX 4000 Ada** or even CPU workers work. For lowest cost start with the smallest GPU available.
4. Container disk: 10 GB is plenty (image is ~5 GB with the model baked in).
5. Workers: min 0 (scale-to-zero), max 1–3 depending on load.
6. Active workers: 0 unless you need to eliminate cold starts.
7. Idle timeout: 5–30 s.
8. (Optional) Env vars: `MODEL_NAME` to swap models without rebuilding.

## Call the endpoint

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "match",
      "query": "Senior Software Engineer",
      "candidates": ["Software Developer", "Registered Nurse", "Principal Software Engineer"],
      "top_k": 3
    }
  }'
```

### Embed

```json
{
  "input": {
    "task": "embed",
    "texts": ["Software Engineer", "Data Scientist"],
    "normalize": true
  }
}
```

Returns `{ "embeddings": [[...], [...]], "dim": 1024, "count": 2, ... }`.

### Match

```json
{
  "input": {
    "task": "match",
    "query": "Senior Software Engineer",
    "candidates": ["Software Developer", "Lead Backend Engineer", "Registered Nurse"],
    "top_k": 3
  }
}
```

Returns `{ "results": [{ "candidate": "...", "score": 0.87, "index": 1 }, ...] }`. Scores are cosine similarity on L2-normalized embeddings, so they live in `[-1, 1]`.

## Notes

- `normalize_embeddings=True` is the default for both tasks; match scores are cosine similarity.
- For large candidate sets, send `embed` for the corpus once, store the vectors yourself, and do the similarity search client-side. Re-encoding candidates per query (which `match` does) is fine up to a few thousand strings but wasteful at scale.
- To switch models, override `MODEL_NAME` at deploy time **and** rebuild the image so the new model is baked in — otherwise the first request pays the download.
