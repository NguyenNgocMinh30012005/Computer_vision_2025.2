import csv
import json
import time
from pathlib import Path


RUN_NAME = "run_17_light_finetune_decision"


RUN14_HELDOUT = [
    {"num_views": 2, "confidence_fscore": 0.425165987570557, "oarh_fscore": 0.37095019351736164, "gated_fscore": 0.425165987570557},
    {"num_views": 3, "confidence_fscore": 0.6029141310268558, "oarh_fscore": 0.47545191168144935, "gated_fscore": 0.6029141310268558},
    {"num_views": 4, "confidence_fscore": 0.6763275638249289, "oarh_fscore": 0.6738369367343314, "gated_fscore": 0.6738369367343314},
    {"num_views": 5, "confidence_fscore": 0.5900714239273983, "oarh_fscore": 0.5703990126573403, "gated_fscore": 0.5900714239273983},
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

    rows = []
    for row in RUN14_HELDOUT:
        best_learned = max(row["oarh_fscore"], row["gated_fscore"])
        delta_vs_conf = best_learned - row["confidence_fscore"]
        rows.append(
            {
                "run": RUN_NAME,
                "num_views": row["num_views"],
                "confidence_fscore": row["confidence_fscore"],
                "best_learned_fscore": best_learned,
                "delta_best_learned_vs_confidence": delta_vs_conf,
                "should_finetune": delta_vs_conf > 0.005,
                "decision": "skip_backbone_finetune" if delta_vs_conf <= 0.005 else "allow_light_finetune",
            }
        )

    allow = any(r["should_finetune"] for r in rows)
    decision = {
        "run": RUN_NAME,
        "global_decision": "allow_light_finetune" if allow else "skip_backbone_finetune",
        "reason": (
            "Run 14 did not show a reliable held-out improvement from OARH/gated learned reliability. "
            "Skipping MV-DUSt3R+ decoder fine-tuning avoids overfitting and keeps the final policy honest."
        ),
        "runtime_seconds": time.time() - started,
    }
    write_csv(out_dir / "fine_tune_decision.csv", rows)
    (out_dir / "run_config.json").write_text(json.dumps(decision, indent=2))
    print("Run 17 fine-tune decision:")
    print(decision)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
