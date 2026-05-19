"""Run the categorizer benchmark against the live RunPod endpoint.

Reports:
  - Per-call latency (client wall, RunPod delayTime, executionTime)
  - Clustering result at multiple thresholds (headline 0.70)
  - Comparison vs Mustafa's prior categorizer

Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID env vars, or fall back to defaults.
"""

import os
import statistics
import sys
import time
from itertools import combinations

import numpy as np
import requests

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "u0m98xy0mqu05v")
API_KEY = os.environ.get("RUNPOD_API_KEY")
if not API_KEY:
    print("ERROR: set RUNPOD_API_KEY env var", file=sys.stderr)
    sys.exit(1)

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
THRESHOLDS = [0.77, 0.70, 0.65, 0.60]
HEADLINE_THRESHOLD = 0.70

RAW = """
Daniel Barcelon       | Senior Data Engineer / Power BI Developer / Senior Data Engineer
Alan Corbeau          | Senior Data Engineer / BI & Analytics Manager / Data & Analytics Lead
Shipra G.             | Analytics Engineer / Data Engineer / Data Science Intern
Marni R.              | Senior Data Engineer / Senior BI & Data Warehouse Developer / Head of Analytics / Senior Data Engineer
Ryan Zhang            | Senior AI & Data Engineer / Data Engineer / Business Intelligence Developer
Willie Pieters        | Data Architect / Engineer (Contractor) / Data Architect / Lead Data Engineer (Contractor)
Henry Upton-Birdsall  | Principal Engineer - Data Platform / Senior Architect - Data Platform / Lead Data Engineer
Savitha Rao           | Senior Data Engineer / Platform Engineer / Data Engineer
Peiran Quan           | Senior Data Engineer / Data Engineer / Data Engineer
David Frost           | Senior Data Engineer / Senior Data Engineer / Data Engineer/Technical Business Analyst
Paravai A.            | Data Engineer / BI & Data Engineer / Data & Insights Analyst
Scott Pedersen        | Senior Engineer - Data / Managing Consultant - Data & Analytics / Consulting Analyst - Business Intelligence
Paul S.               | Senior Data Engineer / Data Lead / Senior Data Engineer
Amit Vajpeyi          | Lead Data Engineer | Data Migration Lead / Data Engineer - Migration Specialist / Data Engineer
David Callander       | Data and SQL Engineer / Senior Data Consultant / Senior Business Analyst
Melissa Foong         | Senior Data Engineer / Business Intel Specialist / Senior Business Intel Analyst
Stephanie L.          | Senior Analytics Engineer / Senior Data Engineer / Solution/Data Engineer
Sri Ram               | Senior Data Engineer / Data Engineer / Platform Engineer - Data
Kevin GAO             | Data Engineer / Product owner / Data and Analytics Engineer / Data Engineer
Craig Peyper          | Senior Data Engineer / Senior Analytics Specialist / Data Platform Manager
"""


def parse_candidates() -> list[tuple[str, list[str]]]:
    out = []
    for line in RAW.strip().splitlines():
        name, titles_str = line.split("|", 1)
        titles = [t.strip() for t in titles_str.split(" / ") if t.strip()]
        out.append((name.strip(), titles))
    return out


def call_runsync(payload: dict) -> tuple[dict, float]:
    """POST /runsync, return (response_json, wall_clock_seconds)."""
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/runsync", headers=HEADERS, json={"input": payload}, timeout=120)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), wall


def cluster(n: int, sims: np.ndarray, threshold: float) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in combinations(range(n), 2):
        if sims[i, j] >= threshold:
            union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def fmt_stats(label: str, vals: list[float]) -> str:
    if not vals:
        return f"  {label}: (no data)"
    vals_sorted = sorted(vals)

    def pct(p):
        k = max(0, min(len(vals_sorted) - 1, int(round(p * (len(vals_sorted) - 1)))))
        return vals_sorted[k]

    return (
        f"  {label:<24}  n={len(vals):<3}  "
        f"min={min(vals):>7.0f}  p50={pct(0.5):>7.0f}  "
        f"avg={statistics.mean(vals):>7.0f}  p95={pct(0.95):>7.0f}  max={max(vals):>7.0f}  (ms)"
    )


def main():
    candidates = parse_candidates()
    all_titles = sorted({t for _, titles in candidates for t in titles})
    print(f"Endpoint: {BASE_URL}")
    print(f"Unique titles: {len(all_titles)}")
    print(f"Candidates: {len(candidates)}\n")

    # --- 1. Warm-up ping ---
    print("Warm-up...")
    _, _ = call_runsync({"task": "embed", "texts": ["warmup"]})

    # --- 2. Big-batch embed (the realistic categorizer mode) ---
    print(f"Embedding {len(all_titles)} unique titles in one batch...")
    resp, wall_batch = call_runsync({"task": "embed", "texts": all_titles})
    batch_delay = resp.get("delayTime", 0)
    batch_exec = resp.get("executionTime", 0)
    vectors = np.asarray(resp["output"]["embeddings"], dtype=np.float32)
    title_to_vec = dict(zip(all_titles, vectors))
    dim = resp["output"]["dim"]
    print(f"  dim={dim}  wall={wall_batch*1000:.0f}ms  delayTime={batch_delay}ms  executionTime={batch_exec}ms")
    print(f"  -> per-title cost: {batch_exec/len(all_titles):.1f}ms execution, {wall_batch*1000/len(all_titles):.1f}ms wall")

    # --- 3. Small-batch latency distribution (per-candidate match calls) ---
    print(f"\nPer-candidate match calls ({len(candidates)} requests)...")
    walls, delays, execs = [], [], []
    for name, titles in candidates:
        query = titles[0]
        cands = titles[1:] if len(titles) > 1 else titles
        resp, wall = call_runsync({"task": "match", "query": query, "candidates": cands, "top_k": min(3, len(cands))})
        walls.append(wall * 1000)
        delays.append(resp.get("delayTime", 0))
        execs.append(resp.get("executionTime", 0))

    print(fmt_stats("wall (client)", walls))
    print(fmt_stats("delayTime (RunPod)", delays))
    print(fmt_stats("executionTime (handler)", execs))

    # --- 4. Clustering analysis using the batch embeddings ---
    print(f"\n{'='*72}\nClustering — HEADLINE threshold {HEADLINE_THRESHOLD}\n{'='*72}")
    headline = None
    summaries = []
    for thr in THRESHOLDS:
        total_titles = 0
        total_clusters = 0
        cands_merged = 0
        per_cand = []
        for name, titles in candidates:
            embs = np.stack([title_to_vec[t] for t in titles])
            sims = embs @ embs.T
            cls = cluster(len(titles), sims, thr)
            total_titles += len(titles)
            total_clusters += len(cls)
            if len(cls) < len(titles):
                cands_merged += 1
            per_cand.append((name, titles, cls, sims))
        compression = 1 - total_clusters / total_titles
        summaries.append((thr, total_titles, total_clusters, compression, cands_merged))
        if thr == HEADLINE_THRESHOLD:
            headline = per_cand

    for name, titles, cls, sims in headline:
        tag = "MERGE" if len(cls) < len(titles) else "----"
        print(f"\n[{tag}] {name}   ({len(titles)} -> {len(cls)})")
        for cl in cls:
            if len(cl) == 1:
                print(f"   - {titles[cl[0]]}")
            else:
                print(f"   - MERGED:")
                for idx in cl:
                    print(f"       * {titles[idx]}")

    print(f"\n{'='*72}\nThreshold sweep\n{'='*72}")
    print(f"{'thr':<6} {'titles':>7} {'clusters':>9} {'compression':>12} {'cands_merged':>13} {'no_merge':>9}")
    for thr, tt, tc, cp, cm in summaries:
        marker = "  <-- headline" if thr == HEADLINE_THRESHOLD else ""
        print(f"{thr:<6} {tt:>7} {tc:>9} {cp*100:>11.1f}% {cm:>13} {len(candidates)-cm:>9}{marker}")

    print(f"\nPrior categorizer (PR #21, threshold 0.77): 60 -> 48 (20% compression), 9/20 candidates with zero merges")

    # --- 5. Final timing summary ---
    print(f"\n{'='*72}\nRunPod response-time summary\n{'='*72}")
    print(f"  Big-batch embed ({len(all_titles)} titles):")
    print(f"    wall = {wall_batch*1000:.0f} ms   delayTime = {batch_delay} ms   executionTime = {batch_exec} ms")
    print(f"  Per-candidate match ({len(candidates)} requests):")
    print(fmt_stats("    wall (client)", walls))
    print(fmt_stats("    delayTime (RunPod)", delays))
    print(fmt_stats("    executionTime (handler)", execs))


if __name__ == "__main__":
    main()
