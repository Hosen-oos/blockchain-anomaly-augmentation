"""Label/time audit and non-graph baseline experiment."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import build_graphs, load_edges
from run_rolling_ablation import rolling_splits


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def period_from_timestamp(timestamp: int, unit: str = "month") -> str:
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return date.strftime("%Y-%m" if unit == "month" else "%Y-Q" + str((date.month - 1) // 3 + 1))


def edge_audit(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(lambda: {"n": 0, "phishing": 0, "benign": 0, "categories": Counter(), "methods": Counter()})
    for row in rows:
        bucket = buckets[period_from_timestamp(int(row.get("timestamp") or 0))]
        bucket["n"] += 1
        label = row.get("label", "benign")
        bucket[label] += 1
        bucket["categories"][row.get("category", "other")] += 1
        bucket["methods"][row.get("input_method", "other")] += 1
    result = []
    for period in sorted(buckets):
        bucket = buckets[period]
        result.append({"period": period, "n": bucket["n"], "phishing": bucket["phishing"], "benign": bucket["benign"],
                       "phishing_rate": bucket["phishing"] / max(bucket["n"], 1),
                       "top_phishing_categories": bucket["categories"].most_common(8),
                       "top_methods": bucket["methods"].most_common(8)})
    return result


def graph_feature_drift(train: list, test: list) -> list[dict]:
    train_matrix = np.stack([graph.graph_features for graph in train])
    test_matrix = np.stack([graph.graph_features for graph in test])
    rows = []
    for index in range(train_matrix.shape[1]):
        train_mean = float(train_matrix[:, index].mean())
        test_mean = float(test_matrix[:, index].mean())
        train_std = float(train_matrix[:, index].std())
        pooled = max((train_matrix[:, index].std() + test_matrix[:, index].std()) / 2.0, 1e-6)
        rows.append({"feature_index": index, "train_mean": train_mean, "test_mean": test_mean,
                     "mean_difference": float(test_mean - train_mean), "standardized_difference": float((test_mean - train_mean) / pooled),
                     "train_std": train_std, "test_std": float(test_matrix[:, index].std())})
    return rows


def metrics(labels, scores, threshold=0.5) -> dict:
    predictions = (np.asarray(scores) >= threshold).astype(int)
    result = {"n": len(labels), "positive": int(sum(labels)), "positive_rate": float(np.mean(labels)),
              "threshold": float(threshold), "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.0,
              "pr_auc": float(average_precision_score(labels, scores)),
              "precision": float(precision_score(labels, predictions, zero_division=0)),
              "recall": float(recall_score(labels, predictions, zero_division=0)),
              "f1": float(f1_score(labels, predictions, zero_division=0))}
    return result


def threshold_at_fpr(model, valid, target_fpr):
    labels = np.asarray([graph.label for graph in valid])
    scores = model.predict_proba(np.stack([graph.graph_features for graph in valid]))[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eligible = np.where(fpr <= target_fpr)[0]
    if not len(eligible):
        return 0.5
    index = eligible[int(np.argmax(tpr[eligible]))]
    return float(thresholds[index]) if np.isfinite(thresholds[index]) else 1.0


def balanced_diagnostic(test: list, seed: int) -> list:
    positive = [graph for graph in test if graph.label == 1]
    negative = [graph for graph in test if graph.label == 0]
    if not positive or not negative:
        return test
    rng = np.random.default_rng(seed)
    sample_count = min(len(negative), len(positive))
    selected_positive = list(rng.choice(positive, size=sample_count, replace=False))
    selected_negative = list(rng.choice(negative, size=sample_count, replace=False))
    return selected_positive + selected_negative


def main() -> None:
    config = load_config(ROOT / "configs" / "audit_experiment.json")
    output_dir = (ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_edges((ROOT / config["edge_file"]).resolve())
    graphs = build_graphs(rows, max_neighbors=int(config["max_neighbors"]), max_edges=int(config["max_edges_per_graph"]))
    results = {"data": {"raw_rows": len(rows), "graphs": len(graphs), "edge_monthly_audit": edge_audit(rows)}, "windows": []}
    for window_index, split in enumerate(rolling_splits(graphs, config["windows"])):
        train, valid, test = split["train"], split["valid"], split["test"]
        x_train = np.stack([graph.graph_features for graph in train])
        y_train = np.asarray([graph.label for graph in train])
        x_valid = np.stack([graph.graph_features for graph in valid])
        y_valid = np.asarray([graph.label for graph in valid])
        x_test = np.stack([graph.graph_features for graph in test])
        y_test = np.asarray([graph.label for graph in test])
        balanced = balanced_diagnostic(test, int(config["seed"]) + window_index)
        x_balanced = np.stack([graph.graph_features for graph in balanced])
        y_balanced = np.asarray([graph.label for graph in balanced])
        window = {"name": split["name"], "label_counts": {"train": Counter(y_train.tolist()), "valid": Counter(y_valid.tolist()), "test": Counter(y_test.tolist()), "balanced_test": Counter(y_balanced.tolist())},
                  "drift": graph_feature_drift(train, test), "models": {}}
        model_specs = [
            ("logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", random_state=int(config["seed"]))),),
            ("random_forest", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=int(config["seed"]), n_jobs=-1, min_samples_leaf=3)),
        ]
        for model_name, model in model_specs:
            model.fit(x_train, y_train)
            valid_scores = model.predict_proba(x_valid)[:, 1]
            test_scores = model.predict_proba(x_test)[:, 1]
            balanced_scores = model.predict_proba(x_balanced)[:, 1]
            model_result = {"natural_test": metrics(y_test, test_scores), "balanced_test": metrics(y_balanced, balanced_scores), "thresholds": {}}
            for target_fpr in config["target_fprs"]:
                threshold = threshold_at_fpr(model, valid, float(target_fpr))
                model_result["thresholds"][str(target_fpr)] = {"threshold": threshold,
                    "valid": metrics(y_valid, valid_scores, threshold),
                    "natural_test": metrics(y_test, test_scores, threshold),
                    "balanced_test": metrics(y_balanced, balanced_scores, threshold)}
            window["models"][model_name] = model_result
        results["windows"].append(window)
    (output_dir / "audit_baseline_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=lambda value: int(value)), encoding="utf-8")
    print(json.dumps({"windows": [window["name"] for window in results["windows"]], "months": len(results["data"]["edge_monthly_audit"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

