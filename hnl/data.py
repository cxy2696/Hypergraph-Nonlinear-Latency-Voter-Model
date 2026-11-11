"""Dataset → hypergraph pipeline for the Kaggle sentiment sample.

Construction follows the paper: users are nodes, a hashtag induces a
hyperedge containing every user who used it, and mean engagement
(``Retweets + Likes``) provides the node weights ``α_i``.

Version 2 makes three previously implicit steps explicit:

* labels are whitespace-normalised before valence mapping;
* hashtag groups are de-duplicated and normalised (``#Serenity`` and
  ``#serenity`` are the same group);
* valence mapping is documented and selectable (``valence``, ``paper``,
  ``strict``), see :mod:`hnl.lexicon`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .hypergraph import Hypergraph
from .lexicon import AMBIGUOUS, NEGATIVE, POSITIVE, classify, normalize_label

MAPPINGS = ("valence", "paper", "strict")


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_hashtag(tag: object) -> str:
    """Case-fold and strip the leading ``#`` from a hashtag token."""
    return _clean(tag).casefold().lstrip("#").strip()


def read_posts(path: str | Path, fmt: Optional[str] = None) -> pd.DataFrame:
    """Read CSV, Parquet or line-delimited JSON posts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = (fmt or p.suffix.lstrip(".")).lower()
    if suffix == "csv":
        df = pd.read_csv(p)
    elif suffix == "parquet":
        df = pd.read_parquet(p)
    elif suffix in ("json", "jsonl"):
        df = pd.read_json(p, lines=True)
    else:
        raise ValueError(f"unsupported format {suffix!r}; use csv, parquet or json")
    return df


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DatasetBundle:
    """Hypergraph plus per-user metadata assembled from a post table."""

    hypergraph: Hypergraph
    labels: np.ndarray
    valence: np.ndarray  # 1 positive, 0 negative, -1 unmapped
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def known_mask(self) -> np.ndarray:
        return self.valence >= 0

    def subgraph(self, nodes: Sequence[int]) -> "DatasetBundle":
        """Induced bundle on ``nodes`` (used for the largest-component analysis)."""
        nodes = np.asarray(sorted(set(int(x) for x in nodes)), dtype=np.int64)
        hg = self.hypergraph.subgraph(nodes)
        lookup = {str(label): int(value) for label, value in zip(self.labels, self.valence)}
        valence = np.array([lookup.get(str(label), -1) for label in hg.labels], dtype=np.int64)
        metadata = dict(self.metadata)
        metadata["scope"] = f"subgraph:{nodes.size}"
        metadata["hypergraph"] = hg.summary()
        return DatasetBundle(
            hypergraph=hg, labels=np.asarray(hg.labels), valence=valence, metadata=metadata
        )

    def opinions_for(
        self, init: str = "random", rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Initial opinions for the engine.

        ``init="data"`` uses mapped sentiment and fills unmapped users by
        drawing from the empirical positive share of the mapped users, so the
        initial balance is preserved.  ``init="random"`` uses a fair coin.
        """
        rng = rng or np.random.default_rng(0)
        n = self.hypergraph.n_nodes
        if init == "random":
            return rng.integers(0, 2, size=n, dtype=np.int64)
        if init != "data":
            raise ValueError("init must be 'random' or 'data'")
        known = self.known_mask
        if not known.any():
            raise ValueError("no user has a mapped sentiment; cannot initialise from data")
        opinions = np.empty(n, dtype=np.int64)
        opinions[known] = self.valence[known].astype(np.int64)
        share = float(self.valence[known].mean())
        n_unknown = int((~known).sum())
        opinions[~known] = (rng.random(n_unknown) < share).astype(np.int64)
        return opinions


def _post_valence(labels: pd.Series, mapping: str) -> np.ndarray:
    normalized = labels.map(normalize_label)
    if mapping == "paper":
        # The paper's rule: Positive -> 1, everything else -> 0.
        return np.where(normalized == "positive", 1, 0)
    if mapping == "strict":
        return np.array(
            [1 if x == "positive" else 0 if x == "negative" else -1 for x in normalized],
            dtype=np.int64,
        )
    if mapping == "valence":
        return np.array([classify(x) if classify(x) is not None else -1 for x in normalized], dtype=np.int64)
    raise ValueError(f"mapping must be one of {MAPPINGS}")


def build_dataset(
    path: str | Path,
    *,
    mapping: str = "valence",
    scope: str = "active",
    min_edge_size: int = 2,
    weight_floor: float = 1e-3,
    user_col: str = "User",
    sentiment_col: str = "Sentiment",
    hashtags_col: str = "Hashtags",
    retweet_col: str = "Retweets",
    like_col: str = "Likes",
    platform_col: str = "Platform",
    fmt: Optional[str] = None,
) -> DatasetBundle:
    """Build the hashtag hypergraph and per-user valence.

    ``scope="active"`` keeps only users that belong to at least one hyperedge
    of size >= 2 (the population that can actually change opinion).
    ``scope="full"`` keeps isolated users too; they are frozen by the engine.
    """
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}")
    if scope not in ("active", "full"):
        raise ValueError("scope must be 'active' or 'full'")
    if min_edge_size < 2:
        raise ValueError("min_edge_size must be at least 2")

    path = Path(path)
    df = read_posts(path, fmt).copy()
    required = [user_col, sentiment_col, hashtags_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    users = df[user_col].map(_clean)
    keep_row = users.ne("")
    df = df.loc[keep_row].reset_index(drop=True)
    users = users.loc[keep_row].reset_index(drop=True)

    user_order: List[str] = list(dict.fromkeys(users.tolist()))
    user_index = {u: i for i, u in enumerate(user_order)}
    n_all = len(user_order)

    hashtags = df[hashtags_col].map(_clean)
    tag_users: Dict[str, set] = {}
    tag_counts: Counter = Counter()
    for user, raw in zip(users.tolist(), hashtags.tolist()):
        seen = set()
        for token in raw.split():
            tag = normalize_hashtag(token)
            if not tag or tag in seen:
                continue
            seen.add(tag)
            tag_users.setdefault(tag, set()).add(user_index[user])
            tag_counts[tag] += 1

    raw_edges = list(tag_users.items())
    dropped_small = [t for t, s in raw_edges if len(s) < min_edge_size]
    edges_all = [(t, sorted(s)) for t, s in raw_edges if len(s) >= min_edge_size]
    seen_edges = set()
    edges_unique = []
    duplicate_edges = 0
    for tag, members in edges_all:
        key = tuple(members)
        if key in seen_edges:
            duplicate_edges += 1
            continue
        seen_edges.add(key)
        edges_unique.append(members)
    if not edges_unique:
        raise ValueError("no hyperedge of the requested minimum size was found")

    covered = sorted({i for members in edges_unique for i in members})
    if scope == "active":
        kept = covered
        relabel = {old: new for new, old in enumerate(kept)}
        edges = [
            [relabel[i] for i in members]
            for members in edges_unique
            if sum(i in relabel for i in members) >= min_edge_size
        ]
        kept_indices = list(kept)
    else:
        kept_indices = list(range(n_all))
        edges = edges_unique

    kept_labels = np.array([user_order[i] for i in kept_indices], dtype=object)

    # node weights: mean engagement per user, normalised to max = 1
    engagement = pd.Series(0.0, index=df.index)
    for col in (retweet_col, like_col):
        if col in df.columns:
            engagement = engagement + pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    per_user = engagement.groupby(users).mean()
    raw_weights = np.array(
        [float(per_user.get(u, 0.0)) for u in kept_labels], dtype=float
    )
    max_w = float(raw_weights.max()) if raw_weights.size else 0.0
    if max_w > 0:
        weights = raw_weights / max_w
    else:  # pragma: no cover - datasets with no engagement
        weights = np.ones_like(raw_weights)
    weights = np.maximum(weights, weight_floor)

    # per-user valence (majority over posts, ties resolved to negative)
    post_val = _post_valence(df[sentiment_col], mapping)
    by_user: Dict[str, List[int]] = defaultdict(list)
    for user, value in zip(users.tolist(), post_val.tolist()):
        if value >= 0:
            by_user[user].append(int(value))
    valence = np.full(len(kept_labels), -1, dtype=np.int64)
    for k, user in enumerate(kept_labels):
        values = by_user.get(str(user))
        if not values:
            continue
        share = float(np.mean(values))
        valence[k] = 1 if share > 0.5 else 0 if share < 0.5 else 0

    hg = Hypergraph(
        n_nodes=len(kept_labels),
        edges=edges,
        weights=weights,
        labels=kept_labels,
    )

    known = valence >= 0
    platform_counts: Dict[str, int] = {}
    if platform_col in df.columns:
        platform_counts = {
            str(k): int(v)
            for k, v in df[platform_col].map(_clean).value_counts().items()
        }
    label_counts = Counter(normalize_label(x) for x in df[sentiment_col].tolist())
    valence_post_counts = {
        "positive": int((post_val == 1).sum()),
        "negative": int((post_val == 0).sum()),
        "unmapped": int((post_val == -1).sum()),
    }

    metadata: Dict[str, Any] = {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": int(len(df)),
        "mapping": mapping,
        "scope": scope,
        "min_edge_size": int(min_edge_size),
        "n_users_total": int(n_all),
        "n_users_kept": int(len(kept_labels)),
        "n_edges": int(hg.n_edges),
        "n_edges_dropped_small": len(dropped_small),
        "n_duplicate_edges_removed": int(duplicate_edges),
        "n_hashtag_groups": int(len(tag_users)),
        "valence_post_counts": valence_post_counts,
        "n_users_with_valence": int(known.sum()),
        "n_users_unmapped": int((~known).sum()),
        "positive_share_mapped": float(valence[known].mean()) if known.any() else float("nan"),
        "platform_counts": platform_counts,
        "top_hashtags": [[t, int(c)] for t, c in tag_counts.most_common(10)],
        "hashtag_group_size_counts": {
            str(size): int(count)
            for size, count in sorted(Counter(len(s) for s in tag_users.values()).items())
        },
        "n_distinct_labels": int(len(label_counts)),
        "n_distinct_labels_mapped": int(
            len(label_counts) - sum(1 for k in label_counts if normalize_label(k) in AMBIGUOUS)
        ),
        "hypergraph": hg.summary(),
    }
    return DatasetBundle(
        hypergraph=hg,
        labels=kept_labels,
        valence=valence,
        metadata=metadata,
    )


def dataset_report(bundle: DatasetBundle) -> str:
    """Human-readable structural report (Markdown)."""
    m = bundle.metadata
    s = m["hypergraph"]
    lines = [
        "# Dataset structure",
        "",
        f"- file: `{m['path']}` (sha256 `{m['sha256'][:16]}…`, {m['bytes']} bytes)",
        f"- rows: {m['rows']}, users: {m['n_users_total']} "
        f"(kept: {m['n_users_kept']}, scope={m['scope']})",
        f"- hyperedges: {m['n_edges']} (dropped size<{m['min_edge_size']}: "
        f"{m['n_edges_dropped_small']}, duplicate groups: {m['n_duplicate_edges_removed']})",
        f"- coverage: {s['coverage']:.1%}; isolated nodes: {s['isolated_nodes']}",
        f"- components: {s['n_components']}; largest component: "
        f"{s['largest_component']} nodes ({s['largest_component_share']:.1%})",
        f"- mean edge size: {s['mean_edge_size']:.2f}; max edge size: {s['max_edge_size']}",
        f"- mean degree (covered): {s['mean_degree_covered']:.2f}",
        f"- sentiment mapping `{m['mapping']}`: "
        f"{m['valence_post_counts']['positive']} positive / "
        f"{m['valence_post_counts']['negative']} negative / "
        f"{m['valence_post_counts']['unmapped']} unmapped posts",
        f"- users with valence: {m['n_users_with_valence']} "
        f"(unmapped: {m['n_users_unmapped']})",
        "",
        "## Top hashtags",
        "",
        "| hashtag | users |",
        "|---|---:|",
    ]
    for tag, count in m["top_hashtags"]:
        lines.append(f"| #{tag} | {count} |")
    return "\n".join(lines) + "\n"

