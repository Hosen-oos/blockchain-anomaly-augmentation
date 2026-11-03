"""Rule/statistical candidate generation and validation for phishing subgraphs."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from data import GraphSample


@dataclass
class ValidationResult:
    structure_score: float
    temporal_score: float
    behavior_score: float
    distribution_score: float
    quality_score: float
    hard_valid: bool
    reasons: list[str]


def _quantile(values: list[float], q: float, default: float) -> float:
    return float(np.quantile(values, q)) if values else default


def _range(values: list[float], low_q: float, high_q: float, default_low: float, default_high: float) -> tuple[float, float]:
    return (_quantile(values, low_q, default_low), _quantile(values, high_q, default_high))


def _sample_from_distribution(rng: np.random.Generator, values: list[float], fallback: float) -> float:
    return float(rng.choice(values)) if values else fallback


def generate_candidates(train_phishing: list[GraphSample], count_per_graph: int, seed: int = 42) -> list[GraphSample]:
    if not train_phishing:
        return []
    rng = np.random.default_rng(seed)
    edge_counts = [max(graph.edge_count, 1) for graph in train_phishing]
    neighbor_counts = [max(graph.neighbor_count, 1) for graph in train_phishing]
    graph_features = np.stack([graph.graph_features for graph in train_phishing])
    candidates: list[GraphSample] = []
    serial = 0
    for source in train_phishing:
        for _ in range(count_per_graph):
            edge_count = int(np.clip(round(_sample_from_distribution(rng, edge_counts, source.edge_count)), 1, 64))
            neighbor_count = int(np.clip(round(_sample_from_distribution(rng, neighbor_counts, source.neighbor_count)), 1, 32))
            sampled_features = graph_features[int(rng.integers(0, len(graph_features)))].copy()
            sampled_features[0] = np.log1p(edge_count)
            sampled_features[1] = np.log1p(neighbor_count)
            sampled_features[2] = float(rng.beta(2.0, 6.0))
            sampled_features[3] = float(rng.beta(6.0, 2.0))
            sampled_features[4] = float(np.exp(rng.normal(np.log(max(source.graph_features[4], 1.0)), 0.35)))
            sampled_features[5] = np.log1p(edge_count / max(sampled_features[4], 1.0))
            sampled_features[6:12] = np.clip(sampled_features[6:12] + rng.normal(0, 0.025, 6), 0, 1)
            sampled_features[12] = float(np.clip(neighbor_count / max(edge_count, 1), 0, 1))
            node_count = neighbor_count + 1
            node_features = np.zeros((node_count, 10), dtype=np.float32)
            node_features[:, 0] = 0.0
            node_features[0, 0] = 1.0
            node_features[0, 1] = edge_count
            node_features[0, 2] = sampled_features[2]
            node_features[0, 4] = sampled_features[6]
            node_features[0, 8] = sampled_features[5]
            for node_index in range(1, node_count):
                node_features[node_index, 1] = max(1.0, edge_count / neighbor_count)
                node_features[node_index, 2] = float(rng.uniform(0.0, 1.0))
                node_features[node_index, 4] = float(rng.uniform(0.0, 1.0))
                node_features[node_index, 8] = float(rng.uniform(0.0, 1.0))
            edge_index = np.vstack([np.zeros(edge_count, dtype=np.int64), rng.integers(1, node_count, edge_count, dtype=np.int64)])
            edge_features = np.zeros((edge_count, 21), dtype=np.float32)
            edge_features[:, 0] = sampled_features[2]
            edge_features[:, 1] = np.sort(rng.uniform(0.0, 1.0, edge_count))
            edge_features[:, 2] = rng.normal(6.0, 2.0, edge_count)
            candidates.append(GraphSample(f"synthetic_{serial:06d}", 1, "phishing", source.timestamp,
                                           np.asarray([f"synthetic_node_{serial}_{i}" for i in range(node_count)]),
                                           edge_index, node_features, sampled_features, edge_features,
                                           edge_count, neighbor_count, source.first_timestamp, source.last_timestamp))
            serial += 1
    return candidates


class RuleStatisticalValidator:
    def __init__(self, real_phishing: list[GraphSample], low_quantile: float = 0.01, high_quantile: float = 0.99):
        self.reference = real_phishing
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.feature_ranges = [_range([float(graph.graph_features[i]) for graph in real_phishing], low_quantile, high_quantile, 0.0, 1.0)
                               for i in range(13)]
        self.edge_range = _range([graph.edge_count for graph in real_phishing], low_quantile, high_quantile, 1, 64)
        self.neighbor_range = _range([graph.neighbor_count for graph in real_phishing], low_quantile, high_quantile, 1, 32)
        self.duration_range = _range([graph.graph_features[4] for graph in real_phishing], low_quantile, high_quantile, 0.0, 3650.0)

    def validate(self, graph: GraphSample) -> ValidationResult:
        reasons: list[str] = []
        structure_checks = []
        for value, (low, high) in zip(graph.graph_features[:4], self.feature_ranges[:4]):
            structure_checks.append(float(low) <= float(value) <= float(high))
        structure_checks.extend([
            self.edge_range[0] <= graph.edge_count <= self.edge_range[1],
            self.neighbor_range[0] <= graph.neighbor_count <= self.neighbor_range[1],
        ])
        structure_score = float(np.mean(structure_checks))
        if not all(structure_checks):
            reasons.append("structure_out_of_range")
        temporal_checks = [self.duration_range[0] <= graph.graph_features[4] <= self.duration_range[1],
                           0.0 <= graph.graph_features[5] <= 15.0]
        temporal_score = float(np.mean(temporal_checks))
        if not all(temporal_checks):
            reasons.append("temporal_out_of_range")
        behavior_values = graph.graph_features[6:12]
        behavior_reference = np.stack([item.graph_features[6:12] for item in self.reference])
        behavior_distance = float(np.mean(np.abs(behavior_values - np.median(behavior_reference, axis=0))))
        behavior_score = float(np.exp(-behavior_distance * 8.0))
        if behavior_score < 0.35:
            reasons.append("behavior_mismatch")
        reference_features = np.stack([item.graph_features for item in self.reference])
        distribution_distance = float(np.mean(np.abs(graph.graph_features - np.median(reference_features, axis=0))))
        distribution_score = float(np.exp(-distribution_distance * 4.0))
        quality_score = float(0.25 * structure_score + 0.2 * temporal_score + 0.3 * behavior_score + 0.25 * distribution_score)
        hard_valid = bool(all(structure_checks) and all(temporal_checks) and behavior_score >= 0.35)
        return ValidationResult(structure_score, temporal_score, behavior_score, distribution_score, quality_score, hard_valid, reasons)


def validate_candidates(candidates: list[GraphSample], validator: RuleStatisticalValidator) -> tuple[list[GraphSample], list[dict]]:
    accepted: list[GraphSample] = []
    records: list[dict] = []
    for graph in candidates:
        result = validator.validate(graph)
        record = {"center": graph.center, "label": graph.label_name, "edge_count": graph.edge_count,
                  "neighbor_count": graph.neighbor_count, **asdict(result)}
        records.append(record)
        if result.hard_valid:
            accepted.append(graph)
    return accepted, records


def save_validation(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
