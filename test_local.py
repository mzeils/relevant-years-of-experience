"""Local smoke test for the handler. Runs without RunPod.

Usage:
    python test_local.py
"""

import json
from pathlib import Path

from handler import handler

with Path(__file__).with_name("test_input.json").open() as f:
    event = json.load(f)

result = handler(event)

if result.get("task") == "embed":
    embs = result.get("embeddings", [])
    preview = {
        **{k: v for k, v in result.items() if k != "embeddings"},
        "embeddings_preview": [emb[:4] + ["..."] for emb in embs[:2]],
    }
    print(json.dumps(preview, indent=2))
else:
    print(json.dumps(result, indent=2))
