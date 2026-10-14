"""Leakage-safe splitting: group by source identity BEFORE any frame extraction."""
from __future__ import annotations

import random
import re

_CELEBDF = re.compile(r"^(id\d+)(?:_(id\d+))?_\d+$")


def identity_tokens(stem: str, original_stem: str | None = None) -> list:
    """Tokens that tie a file to its source(s). Files sharing any token end up in the same group.

    - DFDC metadata: a fake is tied to its `original` video.
    - Celeb-DF:      id0_0000 (real) / id16_id0_0003 (fake) -> identities {id0}, {id16,id0}
    - FaceForensics++: 000 (real) / 000_003 (fake) -> identities {000}, {000,003}
    - otherwise:     the file itself (limitation: near-copies of one source in different files can leak)
    """
    if original_stem:
        return [f"file:{original_stem}", f"file:{stem}"]  # same namespace as a plain file token
    m = _CELEBDF.match(stem)
    if m:
        return [f"id:{t}" for t in m.groups() if t]
    parts = stem.split("_")
    if all(p.isdigit() and len(p) <= 4 for p in parts) and len(parts) in (1, 2):
        return [f"id:{p}" for p in parts]
    return [f"file:{stem}"]


class _UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_groups(items: list, mode: str = "union") -> dict:
    """Map item index -> group id.

    mode="union": connected components over shared identity tokens (strictest; may merge everything into one
                  giant group on datasets whose fakes cross many identity pairs, e.g. Celeb-DF).
    mode="first": group by each file's first identity token only (weaker, heuristic fallback)."""
    uf = _UnionFind()
    toks = []
    for it in items:
        orig = it.get("original")
        t = identity_tokens(it["stem"], orig.rsplit(".", 1)[0] if orig else None)
        toks.append(t)
        for other in t[1:]:
            uf.union(t[0], other)
        uf.find(t[0])
    if mode == "first":
        return {i: t[0] for i, t in enumerate(toks)}
    return {i: uf.find(t[0]) for i, t in enumerate(toks)}


def assign_splits(group_labels: dict, ratios=(0.7, 0.15, 0.15), seed: int = 42) -> dict:
    """group id -> 'train'|'val'|'test'. Stratified by the set of labels the group contains."""
    rng = random.Random(seed)
    buckets: dict = {}
    for g, labels in group_labels.items():
        buckets.setdefault(tuple(sorted(labels)), []).append(g)
    out: dict = {}
    for key in sorted(buckets):
        gs = sorted(buckets[key])
        rng.shuffle(gs)
        n = len(gs)
        n_val, n_test = round(n * ratios[1]), round(n * ratios[2])
        if n >= 3:
            n_val, n_test = max(1, n_val), max(1, n_test)
        n_train = max(0, n - n_val - n_test)
        for i, g in enumerate(gs):
            out[g] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def largest_group_share(groups: dict) -> float:
    from collections import Counter
    return max(Counter(groups.values()).values()) / len(groups) if groups else 0.0
