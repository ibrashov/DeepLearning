
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Support both direct execution and python -m hw9.part4.part4_2.
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


def show_example(row, kind):
    filename = row["filename"]

    with Image.open(DATA_DIR / filename) as img:
        original = img.convert("RGB")

    processed = RECIPES[chosen](original)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    axes[0].imshow(original)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(processed)
    axes[1].set_title(f"Processed: {chosen}")
    axes[1].axis("off")

    plt.suptitle(f"{kind}: {filename}")
    plt.tight_layout()
    plt.show()

    print("Type:", kind)
    print("Condition:", row["condition"])
    print("Reference:", row["reference"])
    print("Baseline:", repr(row["baseline_pred"]))
    print("Chosen:", repr(row["chosen_pred"]))
    print("Baseline edits:", row["baseline_edits"])
    print("Chosen edits:", row["chosen_edits"])
    print("-" * 40)


def show_three_examples():
    baseline = test[
        test["system"] == "baseline"
    ][[
        "filename", "condition", "reference",
        "prediction", "edits"
    ]].rename(columns={
        "prediction": "baseline_pred",
        "edits": "baseline_edits"
    })

    improved_system = test[
        test["system"] == chosen
    ][[
        "filename", "prediction", "edits"
    ]].rename(columns={
        "prediction": "chosen_pred",
        "edits": "chosen_edits"
    })

    merged = baseline.merge(
        improved_system,
        on="filename",
        validate="one_to_one"
    )

    assert len(merged) == 24

    groups = {
        "improvement": merged[
            merged["chosen_edits"] < merged["baseline_edits"]
        ],
        "unchanged": merged[
            merged["chosen_pred"] == merged["baseline_pred"]
        ],
        "regression": merged[
            merged["chosen_edits"] > merged["baseline_edits"]
        ]
    }

    selected = []
    used_files = set()

    # Try one example of each requested type
    for kind, group in groups.items():
        available = group[
            ~group["filename"].isin(used_files)
        ]

        if available.empty:
            print(f"No {kind} example exists.")
            continue

        row = available.iloc[0]
        selected.append((kind, row))
        used_files.add(row["filename"])

    # If a type is absent, select additional real examples
    for kind, group in groups.items():
        if len(selected) >= 3:
            break

        for _, row in group.iterrows():
            if row["filename"] in used_files:
                continue

            selected.append((kind, row))
            used_files.add(row["filename"])

            if len(selected) >= 3:
                break

    for kind, row in selected:
        show_example(row, kind)


if __name__ == "__main__":
    show_three_examples()
