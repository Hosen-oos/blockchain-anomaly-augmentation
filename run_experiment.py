"""Run the first minimal experiment: statistical candidates, validators, GraphSAGE/GAT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from augmentation import RuleStatisticalValidator, generate_candidates, save_validation, validate_candidates
from data import build_graphs, load_edges, save_graphs_jsonl, split_temporally
from detector import train_model


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    args = parser.parse_args()
    config = load_config(Path(args.config))
    output_dir = (ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    edge_file = (ROOT / config["edge_file"]).resolve()
    rows = load_edges(edge_file)
    graphs = build_graphs(rows, max_neighbors=int(config["max_neighbors"]), max_edges=int(config["max_edges_per_graph"]))
    train, valid, test = split_temporally(graphs, float(config["train_ratio"]), float(config["valid_ratio"]))
    train_phishing = [graph for graph in train if graph.label == 1]
    candidates = generate_candidates(train_phishing, int(config["candidate_multiplier"]), seed=seed)
    validator = RuleStatisticalValidator(train_phishing, float(config["validation_quantile_low"]), float(config["validation_quantile_high"]))
    accepted, validation_records = validate_candidates(candidates, validator)
    save_validation(validation_records, output_dir / "candidate_validation.json")
    save_graphs_jsonl(accepted, output_dir / "accepted_candidates.jsonl")
    summary = {
        "raw_rows": len(rows), "graphs": len(graphs), "train": len(train), "valid": len(valid), "test": len(test),
        "train_phishing": len(train_phishing), "candidates": len(candidates), "accepted_candidates": len(accepted),
        "acceptance_rate": len(accepted) / max(len(candidates), 1),
        "split_timestamps": {"train_last": train[-1].timestamp, "valid_last": valid[-1].timestamp, "test_last": test[-1].timestamp},
    }
    (output_dir / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    all_results = {"data": summary, "models": {}}
    for kind in ("sage", "gat"):
        for setting, training_graphs in (("raw", train), ("validated_augmented", train + accepted)):
            _, result = train_model(training_graphs, valid, test, kind=kind,
                                    hidden_dim=int(config["hidden_dim"]), epochs=int(config["epochs"]),
                                    learning_rate=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
                                    patience=int(config["patience"]), seed=seed, device_name="cpu")
            all_results["models"][f"{kind}_{setting}"] = result
            print(kind, setting, result["test"])
    (output_dir / "metrics.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

