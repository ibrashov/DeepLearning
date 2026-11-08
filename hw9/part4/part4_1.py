
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Support both direct execution and python -m hw9.part4.part4_1.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part3.part3_3 import fix_a, fix_b, score

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / "week9_homework_data"
RESULTS_DIR = BASE.parent / "part3"
RECIPES ={
    "fix_a": fix_a,
    "fix_b": fix_b
}
with (RESULTS_DIR / "chosen_config.json").open(
    "r", encoding="utf-8"
) as f:
    config = json.load(f)

chosen = config["chosen_recipe"]

predictions = pd.read_csv(
    RESULTS_DIR / "predictions.csv",
    keep_default_na=False
)

predictions["prediction"] = (
    predictions["prediction"].fillna("")
)

test = predictions[
    predictions["split"] == "test"
].copy()


def condition_cer(system, condition):
    group = test[
        (test["system"] == system)
        & (test["condition"] == condition)
    ]

    assert len(group) == 6

    result = score(
        group["reference"].tolist(),
        group["prediction"].tolist()
    )

    return result["corpus_cer"] * 100


def draw_chart():
    conditions = sorted(test["condition"].unique())

    baseline_values = [
        condition_cer("baseline", c)
        for c in conditions
    ]

    chosen_values = [
        condition_cer(chosen, c)
        for c in conditions
    ]

    x = list(range(len(conditions)))
    width = 0.35

    left = [v - width / 2 for v in x]
    right = [v + width / 2 for v in x]

    plt.figure(figsize=(9, 5))

    plt.bar(
        left, baseline_values,
        width=width, label="Baseline"
    )

    plt.bar(
        right, chosen_values,
        width=width, label=chosen
    )

    plt.xticks(x, conditions)
    plt.xlabel("Image condition (6 images each)")
    plt.ylabel("Corpus CER (%) - lower is better")
    plt.title("Test CER: Baseline vs Frozen Recipe")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        BASE / "test_cer_by_condition.png",
        dpi=200
    )

    plt.show()

    print("\nTest CER by condition:")
    print(pd.DataFrame({
        "condition": conditions,
        "baseline_cer_percent": baseline_values,
        "chosen_cer_percent": chosen_values
    }).to_string(index=False))


if __name__ == "__main__":
    draw_chart()
