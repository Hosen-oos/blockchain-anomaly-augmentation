"""Augmentation ratio and candidate-quality screening for the scarce-data hypothesis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from augmentation import RuleStatisticalValidator, generate_candidates, validate_candidates
from data import build_graphs, load_edges
from detector import train_model
from run_low_data_augmentation import stratified_subset
from run_rolling_ablation import rolling_splits


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def candidate_quality_groups(train: list, multiplier: int, config: dict, seed: int):
    phishing = [graph for graph in train if graph.label == 1]
    candidates = generate_candidates(phishing, multiplier, seed=seed)
    validator = RuleStatisticalValidator(phishing, float(config["validation_quantile_low"]), float(config["validation_quantile_high"]))
    accepted, records = validate_candidates(candidates, validator)
    structure_time = [candidate for candidate, record in zip(candidates, records)
                      if "structure_out_of_range" not in record["reasons"]
                      and "temporal_out_of_range" not in record["reasons"]]
    return {"none": [], "raw": candidates, "structure_time": structure_time, "full": accepted}, {
        "candidate_count": len(candidates), "structure_time_count": len(structure_time),
        "full_count": len(accepted), "full_acceptance_rate": len(accepted) / max(len(candidates), 1),
        "behavior_rejected": sum("behavior_mismatch" in record["reasons"] for record in records),
    }


def label_counts(graphs: list) -> dict:
    return {"n": len(graphs), "positive": int(sum(graph.label for graph in graphs)),
            "positive_rate": float(np.mean([graph.label for graph in graphs])) if graphs else 0.0}


def main() -> None:
    config = load_config(ROOT / "configs" / "augmentation_ratio_experiment.json")
    output_dir = (ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_edges((ROOT / config["edge_file"]).resolve())
    graphs = build_graphs(rows, max_neighbors=int(config["max_neighbors"]), max_edges=int(config["max_edges_per_graph"]))
    split = rolling_splits(graphs, config["windows"])[0]
    full_train, valid, test = split["train"], split["valid"], split["test"]
    output_path = output_dir / "augmentation_ratio_metrics.json"
    results = {"purpose": "estimate how augmentation ratio and candidate quality affect scarce-data training", "window": split["name"],
               "split": {"train": label_counts(full_train), "valid": label_counts(valid), "test": label_counts(test)}, "runs": []}
    for fraction in config["fractions"]:
        for seed in config["seeds"]:
            train = stratified_subset(full_train, float(fraction), int(seed))
            for multiplier in config["multipliers"]:
                groups, candidate_summary = candidate_quality_groups(train, int(multiplier), config, int(seed) + int(multiplier * 100))
                for quality in config["qualities"]:
                    extra = groups[quality]
                    _, result = train_model(train + extra, valid, test, kind="gat",
                        hidden_dim=int(config["hidden_dim"]), epochs=int(config["epochs"]),
                        learning_rate=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
                        patience=int(config["patience"]), seed=int(seed), device_name=config.get("device", "cpu"))
                    compact = {"roc_auc": result["test"]["roc_auc"], "pr_auc": result["test"]["pr_auc"],
                               "precision": result["test"]["precision"], "recall": result["test"]["recall"], "f1": result["test"]["f1"],
                               "recall_at_fpr_1pct": result["test"]["recall_at_fpr_1pct"],
                               "recall_at_fpr_5pct": result["test"]["recall_at_fpr_5pct"]}
                    results["runs"].append({"fraction": fraction, "seed": seed, "multiplier": multiplier, "quality": quality,
                        "train": label_counts(train), "training_n": len(train) + len(extra),
                        "training_positive": int(sum(graph.label for graph in train + extra)),
                        "candidate_summary": candidate_summary, "test": compact})
                    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"fraction={fraction} seed={seed} multiplier={multiplier} quality={quality} test={compact}", flush=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(results["runs"]), "output": str(output_dir / "augmentation_ratio_metrics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
