"""Static hypergraph container and structural diagnostics.

A hypergraph is a pair ``(V, E)`` with ``V = {0, ..., n-1}`` and
``E = {e_1, ..., e_m}``, ``e_k ⊆ V``, ``|e_k| >= 2``.  Edges are stored as
sorted, duplicate-free integer arrays.  Node weights (the paper's ``α_i``) are
attached to nodes, not to node–edge incidences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _as_edge_array(edge: Iterable[int], n_nodes: int) -> np.ndarray:
    arr = np.asarray(list(edge), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("each hyperedge must be one-dimensional")
    if arr.size and (arr.min() < 0 or arr.max() >= n_nodes):
        raise ValueError(
            f"hyperedge contains a node id outside [0, {n_nodes - 1}]: {arr.tolist()}"
        )
    return np.unique(arr)


@dataclass
class Hypergraph:
    """Immutable-by-convention hypergraph with precomputed incidence arrays."""

    n_nodes: int
    edges: Sequence[Sequence[int]]
    weights: Optional[Sequence[float]] = None
    labels: Optional[Sequence[str]] = None

    # populated in __post_init__
    edge_list: List[np.ndarray] = field(init=False, repr=False)
    node_weights: np.ndarray = field(init=False, repr=False)
    degree: np.ndarray = field(init=False, repr=False)
    edge_matrix: np.ndarray = field(init=False, repr=False)
    edge_mask: np.ndarray = field(init=False, repr=False)
    edge_weights: np.ndarray = field(init=False, repr=False)
    incidence: np.ndarray = field(init=False, repr=False)  # (n, max_degree), -1 padded
    incidence_mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive")
        if self.weights is None:
            weights = np.ones(self.n_nodes, dtype=float)
        else:
            weights = np.asarray(self.weights, dtype=float).ravel()
            if weights.size != self.n_nodes:
                raise ValueError("weights must have one entry per node")
            if not np.all(np.isfinite(weights)):
                raise ValueError("weights must be finite")
            if np.any(weights <= 0):
                raise ValueError("weights must be strictly positive")

        seen = set()
        edge_list: List[np.ndarray] = []
        for raw in self.edges:
            edge = _as_edge_array(raw, self.n_nodes)
            if edge.size < 2:
                continue
            key = tuple(int(x) for x in edge)
            if key in seen:
                continue
            seen.add(key)
            edge_list.append(edge)
        if not edge_list:
            raise ValueError("hypergraph has no hyperedge of size >= 2")

        self.edge_list = edge_list
        self.node_weights = weights
        self.labels = None if self.labels is None else np.asarray(self.labels)

        n = self.n_nodes
        degree = np.zeros(n, dtype=np.int64)
        for edge in edge_list:
            degree[edge] += 1
        self.degree = degree

        max_size = max(edge.size for edge in edge_list)
        m = len(edge_list)
        edge_matrix = np.zeros((m, max_size), dtype=np.int64)
        edge_mask = np.zeros((m, max_size), dtype=bool)
        edge_weights = np.zeros((m, max_size), dtype=float)
        for k, edge in enumerate(edge_list):
            s = edge.size
            edge_matrix[k, :s] = edge
            edge_mask[k, :s] = True
            edge_weights[k, :s] = weights[edge]
        self.edge_matrix = edge_matrix
        self.edge_mask = edge_mask
        self.edge_weights = edge_weights

        max_degree = int(degree.max()) if degree.size else 0
        incidence = np.full((n, max_degree), -1, dtype=np.int64)
        cursor = np.zeros(n, dtype=np.int64)
        for k, edge in enumerate(edge_list):
            for node in edge:
                incidence[node, cursor[node]] = k
                cursor[node] += 1
        self.incidence = incidence
        self.incidence_mask = incidence >= 0

    # ------------------------------------------------------------------ basics
    @property
    def n_edges(self) -> int:
        return len(self.edge_list)

    @property
    def sizes(self) -> np.ndarray:
        return np.asarray([e.size for e in self.edge_list], dtype=np.int64)

    @property
    def covered_nodes(self) -> np.ndarray:
        return np.flatnonzero(self.degree > 0)

    def weights_of(self, nodes: Sequence[int]) -> np.ndarray:
        return self.node_weights[np.asarray(nodes, dtype=np.int64)]

    # ----------------------------------------------------- structural analyses
    def components(self) -> Tuple[np.ndarray, np.ndarray]:
        """Connected components of the 2-section graph.

        Returns ``(labels, sizes)`` where ``labels[i]`` is the component id of
        node ``i`` and ``sizes[c]`` its size.  Component ids are ordered by
        decreasing size.
        """
        parent = np.arange(self.n_nodes, dtype=np.int64)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = int(parent[a])
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for edge in self.edge_list:
            anchor = int(edge[0])
            for node in edge[1:]:
                union(anchor, int(node))

        roots = np.asarray([find(i) for i in range(self.n_nodes)], dtype=np.int64)
        _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        labels = rank[inverse]
        return labels, counts[order]

    def giant_component(self) -> np.ndarray:
        labels, sizes = self.components()
        if sizes.size == 0:
            return np.arange(self.n_nodes)
        return np.flatnonzero(labels == 0)

    def subgraph(self, nodes: Sequence[int]) -> "Hypergraph":
        """Induced subgraph on ``nodes`` (edges keep only retained members)."""
        nodes = np.asarray(sorted(set(int(x) for x in nodes)), dtype=np.int64)
        if nodes.size == 0:
            raise ValueError("empty node set")
        relabel = {int(old): new for new, old in enumerate(nodes)}
        edges = []
        for edge in self.edge_list:
            kept = [relabel[int(x)] for x in edge if int(x) in relabel]
            if len(kept) >= 2:
                edges.append(kept)
        labels = None if self.labels is None else [self.labels[i] for i in nodes]
        return Hypergraph(
            n_nodes=int(nodes.size),
            edges=edges,
            weights=self.node_weights[nodes],
            labels=labels,
        )

    def summary(self) -> Dict[str, float]:
        """Descriptive statistics used in reports and run manifests."""
        labels, sizes = self.components()
        covered = self.covered_nodes
        deg = self.degree[covered] if covered.size else np.array([0])
        total_weight = float(self.node_weights.sum())
        largest = int(sizes[0]) if sizes.size else 0
        return {
            "n_nodes": int(self.n_nodes),
            "n_edges": int(self.n_edges),
            "n_edges_raw": int(self.n_edges),
            "mean_edge_size": float(self.sizes.mean()),
            "min_edge_size": int(self.sizes.min()),
            "max_edge_size": int(self.sizes.max()),
            "n_components": int(sizes.size),
            "n_singleton_components": int((sizes == 1).sum()),
            "largest_component": largest,
            "largest_component_share": float(largest / self.n_nodes),
            "covered_nodes": int(covered.size),
            "coverage": float(covered.size / self.n_nodes),
            "isolated_nodes": int(self.n_nodes - covered.size),
            "mean_degree_covered": float(deg.mean()),
            "max_degree": int(self.degree.max()),
            "mean_node_weight": float(self.node_weights.mean()),
            "total_node_weight": total_weight,
        }

    def describe(self) -> str:  # pragma: no cover - formatting helper
        s = self.summary()
        return (
            f"Hypergraph(n={s['n_nodes']}, m={s['n_edges']}, "
            f"mean|e|={s['mean_edge_size']:.2f}, coverage={s['coverage']:.1%}, "
            f"components={s['n_components']}, "
            f"largest={s['largest_component_share']:.1%})"
        )


def random_hypergraph(
    n_nodes: int,
    n_edges: int,
    size_range: Tuple[int, int],
    rng: np.random.Generator,
    weight_mean: float = 1.0,
    weight_sd: float = 0.2,
    weight_floor: float = 0.1,
) -> Hypergraph:
    """Random hypergraph with uniformly sampled edges and Normal node weights.

    This reproduces the construction of ``random_simulation.py`` (edges sampled
    without replacement, weights ``N(1, 0.2²)`` clipped below at ``0.1``), but
    with an explicit generator.
    """
    lo, hi = int(size_range[0]), int(size_range[1])
    if lo < 2 or hi < lo or hi > n_nodes:
        raise ValueError("size_range must satisfy 2 <= lo <= hi <= n_nodes")
    edges = [
        rng.choice(n_nodes, size=int(rng.integers(lo, hi + 1)), replace=False)
        for _ in range(int(n_edges))
    ]
    weights = np.maximum(
        weight_floor, rng.normal(weight_mean, weight_sd, size=n_nodes)
    )
    return Hypergraph(n_nodes=n_nodes, edges=edges, weights=weights)
