"""Data loading and feature construction for the v1 Ethereum phishing experiment."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

METHODS = ["0x23b872dd", "0x095ea7b3", "0xa9059cbb", "0x", "0x5fba79f5", "0xa8174404", "0x42842e0e", "other"]
CATEGORIES = ["Approve", "Permit", "Wallet function", "Airdrop function", "SetApproveForAll", "Free buy order", "Proxy upgrade", "other"]


def _bucket(value: float, edges: tuple[float, ...]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def _safe_log(value: float) -> float:
    return float(math.log1p(max(value, 0.0)))


@dataclass
class GraphSample:
    center: str
    label: int
    label_name: str
    timestamp: int
    nodes: np.ndarray
    edge_index: np.ndarray
    node_features: np.ndarray
    graph_features: np.ndarray
    edge_features: np.ndarray
    edge_count: int
    neighbor_count: int
    first_timestamp: int
    last_timestamp: int

    def clone_with(self, **updates) -> "GraphSample":
        values = self.__dict__.copy()
        values.update(updates)
        return GraphSample(**values)


def load_edges(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_method(value: str) -> str:
    return value if value in METHODS[:-1] else "other"


def _normalize_category(value: str) -> str:
    return value if value in CATEGORIES[:-1] else "other"


def _edge_vector(row: dict, first_timestamp: int, last_timestamp: int) -> np.ndarray:
    timestamp = int(row.get("timestamp") or 0)
    duration = max(last_timestamp - first_timestamp, 1)
    relative_time = (timestamp - first_timestamp) / duration
    value = _safe_log(float(row.get("value_wei") or 0.0))
    gas = _safe_log(float(row.get("gas") or 0.0))
    gas_price = _safe_log(float(row.get("gas_price") or 0.0))
    method = np.zeros(len(METHODS), dtype=np.float32)
    method[METHODS.index(_normalize_method(row.get("input_method", "")))] = 1.0
    category = np.zeros(len(CATEGORIES), dtype=np.float32)
    category[CATEGORIES.index(_normalize_category(row.get("category", "")))] = 1.0
    direction = 1.0 if row.get("from_address") else 0.0
    return np.concatenate(([direction, relative_time, value, gas, gas_price], method, category)).astype(np.float32)


def build_graphs(rows: Iterable[dict], max_neighbors: int = 32, max_edges: int = 64) -> list[GraphSample]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    labels: dict[str, int] = {}
    label_names: dict[str, str] = {}
    for row in rows:
        label_name = row.get("label", "benign")
        for center in (row.get("from_address", "").lower(), row.get("to_address", "").lower()):
            if not center:
                continue
            grouped[center].append(row)
            current = 1 if label_name == "phishing" else 0
            labels[center] = max(labels.get(center, 0), current)
            label_names[center] = "phishing" if labels[center] else "benign"

    graphs: list[GraphSample] = []
    for center, center_rows in grouped.items():
        center_rows = sorted(center_rows, key=lambda item: int(item.get("timestamp") or 0))
        if not center_rows:
            continue
        timestamps = [int(item.get("timestamp") or 0) for item in center_rows]
        first_timestamp, last_timestamp = min(timestamps), max(timestamps)
        neighbor_counts = Counter()
        for row in center_rows:
            other = row.get("to_address", "").lower() if row.get("from_address", "").lower() == center else row.get("from_address", "").lower()
            if other and other != center:
                neighbor_counts[other] += 1
        selected_neighbors = [node for node, _ in neighbor_counts.most_common(max_neighbors)]
        node_index = {center: 0, **{node: index + 1 for index, node in enumerate(selected_neighbors)}}
        selected_rows = [row for row in center_rows if (row.get("from_address", "").lower() in node_index or row.get("to_address", "").lower() in node_index)][:max_edges]
        node_count = len(node_index)
        node_accumulators = [[] for _ in range(node_count)]
        edge_pairs: list[tuple[int, int]] = []
        edge_vectors: list[np.ndarray] = []
        for row in selected_rows:
            source = row.get("from_address", "").lower()
            target = row.get("to_address", "").lower()
            if source not in node_index or target not in node_index or source == target:
                continue
            source_index, target_index = node_index[source], node_index[target]
            vector = _edge_vector(row, first_timestamp, last_timestamp)
            edge_pairs.append((source_index, target_index))
            edge_vectors.append(vector)
            node_accumulators[source_index].append(vector)
            node_accumulators[target_index].append(vector)
        edge_count = len(edge_pairs)
        edge_dim = 5 + len(METHODS) + len(CATEGORIES)
        node_features = np.zeros((node_count, 10), dtype=np.float32)
        for index, values in enumerate(node_accumulators):
            if values:
                matrix = np.stack(values)
                node_features[index, 0] = 1.0 if index == 0 else 0.0
                node_features[index, 1] = len(values)
                node_features[index, 2] = float(matrix[:, 0].mean())
                node_features[index, 3] = float(matrix[:, 1].mean())
                node_features[index, 4] = float(matrix[:, 2].mean())
                node_features[index, 5] = float(matrix[:, 2].std())
                node_features[index, 6] = float(matrix[:, 3].mean())
                node_features[index, 7] = float(matrix[:, 4].mean())
                node_features[index, 8] = float(np.max(matrix[:, 1]) - np.min(matrix[:, 1]))
                node_features[index, 9] = float(len(values) / max(edge_count, 1))
        edge_features = np.stack(edge_vectors).astype(np.float32) if edge_vectors else np.zeros((0, edge_dim), dtype=np.float32)
        in_count = sum(1 for source, target in edge_pairs if target == 0)
        out_count = sum(1 for source, target in edge_pairs if source == 0)
        method_counts = Counter(_normalize_method(row.get("input_method", "")) for row in selected_rows)
        category_counts = Counter(_normalize_category(row.get("category", "")) for row in selected_rows)
        duration_days = max(last_timestamp - first_timestamp, 0) / 86400.0
        burst_score = edge_count / max(duration_days, 1.0)
        graph_features = np.asarray([
            math.log1p(edge_count), math.log1p(len(selected_neighbors)), in_count / max(edge_count, 1),
            out_count / max(edge_count, 1), duration_days, math.log1p(burst_score),
            method_counts["0x23b872dd"] / max(edge_count, 1), method_counts["0x095ea7b3"] / max(edge_count, 1),
            category_counts["Approve"] / max(edge_count, 1), category_counts["Permit"] / max(edge_count, 1),
            category_counts["Wallet function"] / max(edge_count, 1), category_counts["Airdrop function"] / max(edge_count, 1),
            len(set(row.get("to_address", "").lower() for row in selected_rows)) / max(edge_count, 1),
        ], dtype=np.float32)
        graphs.append(GraphSample(center, labels.get(center, 0), label_names.get(center, "benign"), last_timestamp,
                                  np.asarray(list(node_index.keys())), np.asarray(edge_pairs, dtype=np.int64).T if edge_pairs else np.zeros((2, 0), dtype=np.int64),
                                  node_features, graph_features, edge_features, edge_count, len(selected_neighbors), first_timestamp, last_timestamp))
    return graphs


def split_temporally(graphs: list[GraphSample], train_ratio: float, valid_ratio: float) -> tuple[list[GraphSample], list[GraphSample], list[GraphSample]]:
    ordered = sorted(graphs, key=lambda graph: graph.timestamp)
    train_end = max(1, int(len(ordered) * train_ratio))
    valid_end = max(train_end + 1, int(len(ordered) * (train_ratio + valid_ratio)))
    return ordered[:train_end], ordered[train_end:valid_end], ordered[valid_end:]


def save_graphs_jsonl(graphs: Iterable[GraphSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for graph in graphs:
            handle.write(json.dumps({"center": graph.center, "label": graph.label_name, "timestamp": graph.timestamp,
                                     "edge_count": graph.edge_count, "neighbor_count": graph.neighbor_count,
                                     "graph_features": graph.graph_features.tolist()}, ensure_ascii=False) + "\n")
