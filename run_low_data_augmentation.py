"""Low-data augmentation experiment: does validation-filtered synthesis help complex models?"""
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
from run_rolling_ablation import rolling_splits


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def make_full_candidates(train: list, config: dict, seed: int):
    phishing = [graph for graph in train if graph.label == 1]
    candidates = generate_candidates(phishing, int(config["candidate_multiplier"]), seed=seed)
    validator = RuleStatisticalValidator(phishing, float(config["validation_quantile_low"]), float(config["validation_quantile_high"]))
    accepted, records = validate_candidates(candidates, validator)
    return accepted, {"phishing_train": len(phishing), "candidates": len(candidates), "accepted": len(accepted),
                      "acceptance_rate": len(accepted) / max(len(candidates), 1),
                      "behavior_rejected": sum("behavior_mismatch" in item["reasons"] for item in records)}


def label_counts(graphs: list) -> dict:
    return {"n": len(graphs), "positive": int(sum(graph.label for graph in graphs)),
            "positive_rate": float(np.mean([graph.label for graph in graphs])) if graphs else 0.0}


def stratified_subset(graphs: list, fraction: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    positive = [graph for graph in graphs if graph.label == 1]
    negative = [graph for graph in graphs if graph.label == 0]
    positive_count = max(1, int(round(len(positive) * fraction))) if positive else 0
    negative_count = max(1, int(round(len(negative) * fraction))) if negative else 0
    selected_positive = list(rng.choice(positive, size=min(positive_count, len(positive)), replace=False))
    selected_negative = list(rng.choice(negative, size=min(negative_count, len(negative)), replace=False))
    return selected_positive + selected_negative

def main() -> None:
    config = load_config(ROOT / "configs" / "low_data_augmentation.json")
    output_dir = (ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_edges((ROOT / config["edge_file"]).resolve())
    graphs = build_graphs(rows, max_neighbors=int(config["max_neighbors"]), max_edges=int(config["max_edges_per_graph"]))
    split = rolling_splits(graphs, config["windows"])[0]
    full_train, valid, test = split["train"], split["valid"], split["test"]
    results = {"purpose": "test whether augmentation compensates for scarce labeled training graphs", "window": split["name"],
               "full_split": {"train": label_counts(full_train), "valid": label_counts(valid), "test": label_counts(test)}, "runs": []}
    for fraction in config["fractions"]:
        train = stratified_subset(full_train, float(fraction), int(config["seed"]) + int(fraction * 1000))
        accepted, candidate_summary = make_full_candidates(train, config, int(config["seed"]) + int(fraction * 1000))
        settings = {"none": [], "full": accepted}
        for kind in config["models"]:
            for setting, extra in settings.items():
                model, result = train_model(train + extra, valid, test, kind=kind,
                    hidden_dim=int(config["hidden_dim"]), epochs=int(config["epochs"]),
                    learning_rate=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
                    patience=int(config["patience"]), seed=int(config["seed"]), device_name=config.get("device", "cpu"))
                results["runs"].append({"fraction": fraction, "train": label_counts(train), "setting": setting, "model": kind,
                    "candidate_summary": candidate_summary, "training_n": len(train) + len(extra),
                    "training_positive": int(sum(graph.label for graph in train + extra)), "result": result})
                print(f"fraction={fraction} model={kind} setting={setting} test={result['test']}")
    (output_dir / "low_data_augmentation_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(results["runs"]), "output": str(output_dir / "low_data_augmentation_metrics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
