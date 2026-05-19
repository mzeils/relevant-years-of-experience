"""Benchmark JobBERT-v2 against Mustafa's 20-candidate categorizer test set.

Uses single-link cosine-similarity clustering at threshold 0.77 (matching
PR #21's threshold). Reports per-candidate clusters, totals, and explicit
checks on the three worst cases Mustafa called out.
"""

import os
import sys
from itertools import combinations

import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = os.environ.get("MODEL_NAME", "TechWolf/JobBERT-v2")
THRESHOLDS = [0.77, 0.70, 0.65, 0.60]
DEFAULT_THRESHOLD = 0.77

# 20 candidates, top-3 most recent roles each (Mustafa's test set).
# Split on " / " (with spaces) — internal slashes inside a single title are kept.
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
        # Note: Amit Vajpeyi has a literal "|" inside his first title.
        # We split on the FIRST "|" only (above), so subsequent | stays in titles.
        titles = [t.strip() for t in titles_str.split(" / ") if t.strip()]
        out.append((name.strip(), titles))
    return out


def cluster(titles: list[str], sims: torch.Tensor, threshold: float) -> list[list[int]]:
    """Single-link clustering: union pairs whose similarity >= threshold."""
    n = len(titles)
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


def run_for_threshold(candidates, embeddings_by_title, threshold: float, *, verbose: bool) -> dict:
    total_titles = 0
    total_clusters = 0
    candidates_with_merges = 0
    per_cand = []

    for name, titles in candidates:
        n = len(titles)
        total_titles += n
        embs = torch.stack([embeddings_by_title[t] for t in titles])
        sims = util.cos_sim(embs, embs)
        clusters = cluster(titles, sims, threshold)
        total_clusters += len(clusters)
        merged = n - len(clusters)
        if merged > 0:
            candidates_with_merges += 1
        per_cand.append((name, titles, clusters, sims, merged))

    if verbose:
        print(f"\n{'='*72}\nThreshold = {threshold}\n{'='*72}")
        for name, titles, clusters, sims, merged in per_cand:
            tag = "MERGE" if merged > 0 else "----"
            print(f"\n[{tag}] {name}   ({len(titles)} titles -> {len(clusters)} clusters)")
            for cl in clusters:
                if len(cl) == 1:
                    print(f"   - {titles[cl[0]]}")
                else:
                    print(f"   - MERGED:")
                    for idx in cl:
                        print(f"       * {titles[idx]}")
            # Show off-diagonal max-sim for transparency on near-misses
            if len(titles) > 1 and merged == 0:
                pairs = []
                for i, j in combinations(range(len(titles)), 2):
                    pairs.append((sims[i, j].item(), i, j))
                pairs.sort(reverse=True)
                top = pairs[0]
                print(f"   (highest pairwise sim: {top[0]:.3f}  '{titles[top[1]]}' <-> '{titles[top[2]]}')")

    compression = 1 - total_clusters / total_titles if total_titles else 0
    return {
        "threshold": threshold,
        "total_titles": total_titles,
        "total_clusters": total_clusters,
        "compression": compression,
        "candidates_with_merges": candidates_with_merges,
        "candidates_with_no_merges": len(candidates) - candidates_with_merges,
        "per_cand": per_cand,
    }


def main():
    candidates = parse_candidates()
    all_titles = sorted({t for _, titles in candidates for t in titles})
    print(f"Loading {MODEL_NAME}...", file=sys.stderr)
    model = SentenceTransformer(MODEL_NAME)
    print(f"Encoding {len(all_titles)} unique titles...", file=sys.stderr)
    embs = model.encode(all_titles, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    embeddings_by_title = dict(zip(all_titles, embs))

    summaries = []
    for thr in THRESHOLDS:
        verbose = thr == DEFAULT_THRESHOLD
        summary = run_for_threshold(candidates, embeddings_by_title, thr, verbose=verbose)
        summaries.append(summary)

    # Worst-case spot-checks (Mustafa's callouts) at the default threshold.
    print(f"\n{'='*72}\nWorst-case spot checks (threshold {DEFAULT_THRESHOLD})\n{'='*72}")
    for name in ("Amit Vajpeyi", "Ryan Zhang", "Marni R."):
        per = next(s["per_cand"] for s in summaries if s["threshold"] == DEFAULT_THRESHOLD)
        cand = next((c for c in per if c[0] == name), None)
        if not cand:
            continue
        _, titles, clusters, sims, merged = cand
        print(f"\n  {name}: {len(titles)} -> {len(clusters)} ({'MERGED' if merged else 'no merge'})")
        for i, j in combinations(range(len(titles)), 2):
            print(f"    sim {sims[i,j].item():.3f}  '{titles[i]}'  vs  '{titles[j]}'")

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"{'thr':<6} {'titles':>7} {'clusters':>9} {'compression':>12} {'cands_merged':>13} {'cands_no_merge':>15}")
    for s in summaries:
        print(
            f"{s['threshold']:<6} {s['total_titles']:>7} {s['total_clusters']:>9} "
            f"{s['compression']*100:>11.1f}% {s['candidates_with_merges']:>13} {s['candidates_with_no_merges']:>15}"
        )

    print(f"\nMustafa's prior categorizer (PR #21, threshold 0.77):")
    print(f"  60 titles -> 48 clusters  (20.0% compression)   9/20 candidates had zero merges")


if __name__ == "__main__":
    main()
