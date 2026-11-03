"""Pure PyTorch GraphSAGE and GAT classifiers for small ego-subgraphs."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch import nn
from torch.nn import functional as F

from data import GraphSample


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        neighbor_sum = torch.zeros_like(x)
        degree = torch.zeros(x.shape[0], device=x.device)
        if source.numel():
            neighbor_sum.index_add_(0, target, x[source])
            neighbor_sum.index_add_(0, source, x[target])
            degree.index_add_(0, target, torch.ones_like(target, dtype=torch.float32))
            degree.index_add_(0, source, torch.ones_like(source, dtype=torch.float32))
        neighbor_mean = neighbor_sum / degree.clamp_min(1.0).unsqueeze(-1)
        return torch.relu(self.self_linear(x) + self.neighbor_linear(neighbor_mean))


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(2 * out_dim))
        nn.init.xavier_uniform_(self.attention.view(1, -1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        transformed = self.projection(x)
        source, target = edge_index
        if source.numel() == 0:
            return torch.relu(transformed)
        source_all = torch.cat([source, target])
        target_all = torch.cat([target, source])
        scores = F.leaky_relu(torch.cat([transformed[source_all], transformed[target_all]], dim=-1) @ self.attention, 0.2)
        output = torch.zeros_like(transformed)
        for node in range(transformed.shape[0]):
            mask = target_all == node
            if mask.any():
                weights = torch.softmax(scores[mask], dim=0)
                output[node] = (transformed[source_all[mask]] * weights.unsqueeze(-1)).sum(dim=0)
        return torch.relu(output + transformed)


class EgoGraphDetector(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 32, kind: str = "sage"):
        super().__init__()
        layer = GraphSAGELayer if kind.lower() == "sage" else GATLayer
        self.kind = kind.lower()
        self.layer1 = layer(node_dim, hidden_dim)
        self.layer2 = layer(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim + 13, hidden_dim), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden_dim, 1))

    def encode_graph(self, graph: GraphSample, device: torch.device) -> torch.Tensor:
        x = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        center = x[0]
        graph_features = torch.tensor(graph.graph_features, dtype=torch.float32, device=device)
        return torch.cat([center, graph_features], dim=0)

    def forward_graph(self, graph: GraphSample, device: torch.device) -> torch.Tensor:
        return self.classifier(self.encode_graph(graph, device)).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_metric(function, y_true, y_score, default=0.0):
    try:
        return float(function(y_true, y_score))
    except ValueError:
        return default


def predict_scores(model: EgoGraphDetector, graphs: list[GraphSample], device: torch.device) -> tuple[list[int], list[float]]:
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for graph in graphs:
            probability = torch.sigmoid(model.forward_graph(graph, device)).item()
            labels.append(graph.label)
            probabilities.append(probability)
    return labels, probabilities


def select_threshold_at_fpr(model: EgoGraphDetector, graphs: list[GraphSample], device: torch.device, target_fpr: float) -> float:
    labels, probabilities = predict_scores(model, graphs, device)
    try:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
        eligible = np.where(false_positive_rate <= target_fpr)[0]
        if len(eligible):
            best_index = eligible[int(np.argmax(true_positive_rate[eligible]))]
            threshold = float(thresholds[best_index])
            return threshold if np.isfinite(threshold) else 1.0
    except ValueError:
        pass
    return 0.5


def evaluate(model: EgoGraphDetector, graphs: list[GraphSample], device: torch.device, threshold: float = 0.5) -> dict:
    labels, probabilities = predict_scores(model, graphs, device)
    predictions = [int(value >= threshold) for value in probabilities]
    metrics = {
        "n": len(graphs),
        "positive": int(sum(labels)),
        "positive_rate": float(np.mean(labels)) if labels else 0.0,
        "threshold": float(threshold),
        "roc_auc": _safe_metric(roc_auc_score, labels, probabilities),
        "pr_auc": _safe_metric(average_precision_score, labels, probabilities),
        "precision": _safe_metric(precision_score, labels, predictions),
        "recall": _safe_metric(recall_score, labels, predictions),
        "f1": _safe_metric(f1_score, labels, predictions),
    }
    try:
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
        for target_fpr in (0.01, 0.05):
            eligible = true_positive_rate[false_positive_rate <= target_fpr]
            metrics[f"recall_at_fpr_{int(target_fpr * 100)}pct"] = float(eligible.max()) if len(eligible) else 0.0
    except ValueError:
        metrics["recall_at_fpr_1pct"] = 0.0
        metrics["recall_at_fpr_5pct"] = 0.0
    return metrics

def train_model(train_graphs: list[GraphSample], valid_graphs: list[GraphSample], test_graphs: list[GraphSample],
                kind: str, hidden_dim: int, epochs: int, learning_rate: float, weight_decay: float,
                patience: int, seed: int, device_name: str = "cpu") -> tuple[EgoGraphDetector, dict]:
    set_seed(seed)
    device = torch.device(device_name)
    model = EgoGraphDetector(node_dim=10, hidden_dim=hidden_dim, kind=kind).to(device)
    labels = torch.tensor([graph.label for graph in train_graphs], dtype=torch.float32, device=device)
    positive = float(labels.sum().item())
    negative = float(len(labels) - positive)
    pos_weight = torch.tensor([negative / max(positive, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_score = -1.0
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_graphs)))
        random.shuffle(order)
        losses = []
        for index in order:
            graph = train_graphs[index]
            optimizer.zero_grad()
            logit = model.forward_graph(graph, device)
            target = torch.tensor(float(graph.label), device=device)
            loss = criterion(logit.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        valid_metrics = evaluate(model, valid_graphs, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid": valid_metrics})
        if valid_metrics["pr_auc"] > best_score:
            best_score = valid_metrics["pr_auc"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"kind": kind, "epochs_ran": len(history), "best_valid_pr_auc": best_score,
                    "valid": evaluate(model, valid_graphs, device), "test": evaluate(model, test_graphs, device), "history": history}


