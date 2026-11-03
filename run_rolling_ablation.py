"""Rolling-window evaluation and augmentation ablation for the first experiment."""
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


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rolling_splits(graphs: list, specs: list[dict]) -> list[dict]:
    ordered = sorted(graphs, key=lambda graph: graph.timestamp)
    splits = []
    for spec in specs:
        n = len(ordered)
        train_start = int(n * spec["train_start"])
        train_end = int(n * spec["train_end"])
        valid_end = int(n * spec["valid_end"])
        test_end = int(n * spec["test_end"])
        train = ordered[train_start:train_end]
        valid = ordered[train_end:valid_end]
        test = ordered[valid_end:test_end]
        splits.append({"name": spec["name"], "train": train, "valid": valid, "test": test})
    return splits


def candidate_groups(train: list, config: dict, seed: int) -> tuple[dict[str, list], dict]:
    phishing = [graph for graph in train if graph.label == 1]
    candidates = generate_candidates(phishing, int(config["candidate_multiplier"]), seed=seed)
    validator = RuleStatisticalValidator(phishing, float(config["validation_quantile_low"]), float(config["validation_quantile_high"]))
    accepted, records = validate_candidates(candidates, validator)
    structure_time = [candidate for candidate, record in zip(candidates, records)
                      if "structure_out_of_range" not in record["reasons"]
                      and "temporal_out_of_range" not in record["reasons"]]
    return {
        "none": [],
        "raw": candidates,
        "structure_time": structure_time,
        "full": accepted,
    }, {
        "train_phishing": len(phishing),
        "candidate_count": len(candidates),
        "structure_time_count": len(structure_time),
        "full_count": len(accepted),
        "full_acceptance_rate": len(accepted) / max(len(candidates), 1),
        "behavior_rejected": sum("behavior_mismatch" in record["reasons"] for record in records),
    }


def label_summary(graphs: list) -> dict:
    return {"n": len(graphs), "positive": int(sum(graph.label for graph in graphs)),
            "positive_rate": float(np.mean([graph.label for graph in graphs])) if graphs else 0.0,
            "first_timestamp": graphs[0].timestamp if graphs else None,
            "last_timestamp": graphs[-1].timestamp if graphs else None}


def main() -> None:
    config = load_config(ROOT / "configs" / "rolling_experiment.json")
    output_dir = (ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_edges((ROOT / config["edge_file"]).resolve())
    graphs = build_graphs(rows, max_neighbors=int(config["max_neighbors"]), max_edges=int(config["max_edges_per_graph"]))
    results = {"data": {"raw_rows": len(rows), "graphs": len(graphs)}, "windows": []}
    for window_index, split in enumerate(rolling_splits(graphs, config["windows"])):
        train, valid, test = split["train"], split["valid"], split["test"]
        groups, candidate_summary = candidate_groups(train, config, int(config["seed"]) + window_index)
        window_result = {
            "name": split["name"],
            "split": {"train": label_summary(train), "valid": label_summary(valid), "test": label_summary(test)},
            "candidates": candidate_summary,
            "models": {},
        }
        for kind in config["models"]:
            for setting, extra_graphs in groups.items():
                _, model_result = train_model(
                    train + extra_graphs, valid, test, kind=kind,
                    hidden_dim=int(config["hidden_dim"]), epochs=int(config["epochs"]),
                    learning_rate=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
                    patience=int(config["patience"]), seed=int(config["seed"]) + window_index,
                    device_name=config.get("device", "cpu"),
                )
                window_result["models"][f"{kind}_{setting}"] = {
                    "training_n": len(train) + len(extra_graphs),
                    "training_positive": int(sum(graph.label for graph in train + extra_graphs)),
                    "result": model_result,
                }
                print(window_result["name"], kind, setting, model_result["test"])
        results["windows"].append(window_result)
    (output_dir / "rolling_ablation_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"windows": [window["name"] for window in results["windows"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
