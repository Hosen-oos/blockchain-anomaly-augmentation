"""Time-distribution audit and validation-calibrated threshold experiment."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from augmentation import RuleStatisticalValidator, generate_candidates, validate_candidates
from data import build_graphs, load_edges
from detector import evaluate, select_threshold_at_fpr, train_model
from run_rolling_ablation import label_summary, rolling_splits


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def month_audit(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(lambda: {"transactions": 0, "phishing": 0, "benign": 0, "categories": Counter()})
    for row in rows:
        timestamp = int(row.get("timestamp") or 0)
        month = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m")
        bucket = buckets[month]
        bucket["transactions"] += 1
        label = row.get("label", "benign")
        bucket[label] = bucket.get(label, 0) + 1
        bucket["categories"][row.get("category", "other")] += 1
    result = []
    for month in sorted(buckets):
        bucket = buckets[month]
        result.append({"month": month, "transactions": bucket["transactions"],
                       "phishing": bucket.get("phishing", 0), "benign": bucket.get("benign", 0),
                       "phishing_rate": bucket.get("phishing", 0) / max(bucket["transactions"], 1),
                       "top_categories": bucket["categories"].most_common(5)})
    return result


def candidate_groups(train: list, config: dict, seed: int) -> tuple[dict[str, list], dict]:
    phishing = [graph for graph in train if graph.label == 1]
    candidates = generate_candidates(phishing, int(config["candidate_multiplier"]), seed=seed)
    validator = RuleStatisticalValidator(phishing, float(config["validation_quantile_low"]), float(config["validation_quantile_high"]))
    accepted, records = validate_candidates(candidates, validator)
    structure_time = [candidate for candidate, record in zip(candidates, records)
                      if "structure_out_of_range" not in record["reasons"]
                      and "temporal_out_of_range" not in record["reasons"]]
    return {"none": [], "raw": candidates, "structure_time": structure_time, "full": accepted}, {
        "train_phishing": len(phishing), "candidate_count": len(candidates),
        "structure_time_count": len(structure_time), "full_count": len(accepted),
        "full_acceptance_rate": len(accepted) / max(len(candidates), 1),
        "behavior_rejected": sum("behavior_mismatch" in record["reasons"] for record in records),
    }


def main() -> None:
    config = load_config(ROOT / "configs" / "calibration_experiment.json")
    output_dir = (ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_edges((ROOT / config["edge_file"]).resolve())
    graphs = build_graphs(rows, max_neighbors=int(config["max_neighbors"]), max_edges=int(config["max_edges_per_graph"]))
    results = {"data": {"raw_rows": len(rows), "graphs": len(graphs), "monthly_audit": month_audit(rows)}, "windows": []}
    for window_index, split in enumerate(rolling_splits(graphs, config["windows"])):
        train, valid, test = split["train"], split["valid"], split["test"]
        groups, candidate_summary = candidate_groups(train, config, int(config["seed"]) + window_index)
        window_result = {"name": split["name"], "split": {"train": label_summary(train), "valid": label_summary(valid), "test": label_summary(test)}, "candidates": candidate_summary, "models": {}}
        for kind in config["models"]:
            for setting, extra_graphs in groups.items():
                model, base_result = train_model(train + extra_graphs, valid, test, kind=kind,
                    hidden_dim=int(config["hidden_dim"]), epochs=int(config["epochs"]),
                    learning_rate=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
                    patience=int(config["patience"]), seed=int(config["seed"]) + window_index,
                    device_name=config.get("device", "cpu"))
                import torch
                device = torch.device(config.get("device", "cpu"))
                calibrated = {}
                for target_fpr in config["target_fprs"]:
                    threshold = select_threshold_at_fpr(model, valid, device, float(target_fpr))
                    calibrated[str(target_fpr)] = {"threshold": threshold, "valid": evaluate(model, valid, device, threshold), "test": evaluate(model, test, device, threshold)}
                window_result["models"][f"{kind}_{setting}"] = {"base": base_result, "calibrated": calibrated,
                    "training_n": len(train) + len(extra_graphs), "training_positive": int(sum(graph.label for graph in train + extra_graphs))}
                print(window_result["name"], kind, setting, {k: v["test"] for k, v in calibrated.items()})
        results["windows"].append(window_result)
    (output_dir / "calibration_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"windows": [window["name"] for window in results["windows"]], "months": len(results["data"]["monthly_audit"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
