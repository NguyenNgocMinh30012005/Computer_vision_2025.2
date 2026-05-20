import csv
import json
import time
from pathlib import Path


RUN_NAME = "run_18_learned_full_evaluation_summary"


ROWS = [
    {
        "comparison": "Run8 scene0000_00",
        "method": "B0 random default",
        "num_views": 2,
        "fscore": 0.663027330965597,
        "note": "Original baseline from full-pipeline ablation",
    },
    {
        "comparison": "Run8 scene0000_00",
        "method": "Final fixed-threshold view-selection pipeline",
        "num_views": 2,
        "fscore": 0.7385050554564019,
        "note": "Best heuristic final pipeline",
    },
    {
        "comparison": "Run8 scene0000_00",
        "method": "Final fixed-threshold view-selection pipeline",
        "num_views": 3,
        "fscore": 0.8118167774356448,
        "note": "Best heuristic final pipeline",
    },
    {
        "comparison": "Run8 scene0000_00",
        "method": "Final fixed-threshold view-selection pipeline",
        "num_views": 4,
        "fscore": 0.8416150675330538,
        "note": "Best heuristic final pipeline",
    },
    {
        "comparison": "Run8 scene0000_00",
        "method": "Final fixed-threshold view-selection pipeline",
        "num_views": 5,
        "fscore": 0.8189963473865427,
        "note": "Best heuristic final pipeline",
    },
    {
        "comparison": "Run14 heldout scene0000_01",
        "method": "confidence_threshold_val_fixed",
        "num_views": 2,
        "fscore": 0.425165987570557,
        "note": "Held-out confidence baseline for learned extension",
    },
    {
        "comparison": "Run14 heldout scene0000_01",
        "method": "validation_gated_learned_pipeline",
        "num_views": 2,
        "fscore": 0.425165987570557,
        "note": "Gated OARH falls back to confidence",
    },
    {
        "comparison": "Run14 heldout scene0000_01",
        "method": "confidence_threshold_val_fixed",
        "num_views": 3,
        "fscore": 0.6029141310268558,
        "note": "Held-out confidence baseline for learned extension",
    },
    {
        "comparison": "Run14 heldout scene0000_01",
        "method": "validation_gated_learned_pipeline",
        "num_views": 3,
        "fscore": 0.6029141310268558,
        "note": "Gated OARH falls back to confidence",
    },
    {
        "comparison": "Run14 heldout scene0000_01",
        "method": "confidence_threshold_val_fixed",
        "num_views": 4,
        "fscore": 0.6763275638249289,
        "note": "Held-out confidence baseline for learned extension",
    },
    {
        "comparison": "Run14 heldout scene0000_01",
        "method": "validation_gated_learned_pipeline",
        "num_views": 4,
        "fscore": 0.6738369367343314,
        "note": "Gate selected OARH but underperformed confidence slightly",
    },
    {
        "comparison": "Run14 heldout scene0000_01",
        "method": "confidence_threshold_val_fixed",
        "num_views": 5,
        "fscore": 0.5900714239273983,
        "note": "Held-out confidence baseline for learned extension",
    },
    {
        "comparison": "Run14 heldout scene0000_01",
        "method": "validation_gated_learned_pipeline",
        "num_views": 5,
        "fscore": 0.5900714239273983,
        "note": "Gated OARH falls back to confidence",
    },
]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "final_learned_summary.csv", ROWS)
    conclusion = {
        "run": RUN_NAME,
        "final_recommendation": "Use fixed confidence threshold + best view selection as final reconstruction policy.",
        "learned_extension_status": (
            "OARH/gated reliability is not yet a reconstruction win on held-out data. "
            "RSDH remains promising as match-level supervision and should be advanced with MASt3R descriptor/cycle features."
        ),
        "runtime_seconds": time.time() - started,
    }
    (out_dir / "run_config.json").write_text(json.dumps(conclusion, indent=2))
    print("Run 18 final learned evaluation summary:")
    print(conclusion)
    for row in ROWS:
        print(row)


if __name__ == "__main__":
    main()
