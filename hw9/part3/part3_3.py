import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from rapidfuzz.distance import Levenshtein

if __package__:
    from .part3_2 import fix_a, fix_b
else:
    from part3_2 import fix_a, fix_b

DATA_DIR = Path(__file__).resolve().parent.parent / "week9_homework_data"
CSV_FILE = DATA_DIR / "manifest.csv"


def normalize(text):
    return "".join(char for char in text.upper() if not char.isspace())


def score(refs,preds):
    assert len(refs) == len(preds), "Lengths must match"
    total_edits = 0
    reference_chars = 0
    exact_matchs = 0
    for ref, pred in zip(refs, preds):
        ref = normalize(ref)
        pred = normalize(pred)
        reference_chars += len(ref)
        edits = Levenshtein.distance(ref,pred)
        total_edits += edits
        if ref == pred:
            exact_matchs +=1
    if reference_chars == 0:
        raise ValueError("No refecence chars")
    corpur_cer = total_edits/reference_chars
    return{
        "images": len(refs),
        "reference_chars": reference_chars,
        "total_edits": total_edits,
        "corpus_cer": corpur_cer,
        "exact_matches": exact_matchs
    }
def ocr(image):
    import pytesseract
    windows_executable = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if windows_executable.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(windows_executable)
    return pytesseract.image_to_string(image, config = "--psm 7")
def run_system(rows, preprocess =None):
    results = []
    for row in rows:
        image_path = DATA_DIR/row["filename"]
        reference = normalize(row["target"])
        prediction = ""
        error = ""
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            if preprocess is not None:
                image = preprocess(image)
            prediction = normalize(ocr(image))
        except Exception as e:
            error = str(e)
            prediction = ""
        edits = Levenshtein.distance(reference, prediction)
        image_cer = edits/max(1,len(reference))
        results.append({
                "filename": row["filename"],
                "base_id": row["base_id"],
                "condition": row["condition"],
                "reference": reference,
                "prediction":prediction,
                "edits": edits,
                "cer": image_cer,
                "exact":reference == prediction,
                "error":error
        })   
    table =pd.DataFrame(results)
    metrics = score(
        table["reference"].tolist(),
        table["prediction"].tolist()
    )
    return table,metrics
RECIPES ={
    "fix_a": fix_a,
    "fix_b": fix_b
}
BASE = Path(__file__).resolve().parent
CONFIG_FILE = BASE / "chosen_config.json"
def make_summary(table):
    summary = []
    groups = list(table.groupby("condition"))
    groups.append(("OVERALL", table))
    for condition, group in groups:
        metrics = score(
            group["reference"].tolist(),
            group["prediction"].tolist()
        )
        summary.append({
            "condition": condition,
            "images": metrics["images"],
            "reference_chars": metrics["reference_chars"],
            "edits": metrics["total_edits"],
            "corpus_cer": round(metrics["corpus_cer"], 4),
            "exact_matches": metrics["exact_matches"]
        })

    return pd.DataFrame(summary)


def preview_recipes(rows):
    conditions = sorted({r["condition"] for r in rows})

    fig, axes = plt.subplots(
        len(conditions), 3,
        figsize=(12, 3 * len(conditions)),
        squeeze=False
    )

    for i, condition in enumerate(conditions):
        row = next(
            r for r in rows
            if r["condition"] == condition
        )

        with Image.open(DATA_DIR / row["filename"]) as im:
            original = im.convert("RGB")

        images = [
            original,
            RECIPES["fix_a"](original),
            RECIPES["fix_b"](original)
        ]

        for j, image in enumerate(images):
            axes[i][j].imshow(image)
            axes[i][j].set_title(
                f"{condition} - {['Original', 'Fix A', 'Fix B'][j]}"
            )
            axes[i][j].axis("off")

    plt.tight_layout()
    plt.savefig(BASE / "recipe_preview.png", dpi=150)
    plt.show()
    plt.close()


def main():
    if CONFIG_FILE.exists():
        raise FileExistsError(
            "Experiment already frozen. Do not run validation again."
        )

    df = pd.read_csv(CSV_FILE)

    rows = df[
        df["split"] == "validation"
    ].to_dict("records")

    assert len(rows) == 24
    preview_recipes(rows)

    metrics_by_system = {}

    systems = {
        "baseline": None,
        "fix_a": RECIPES["fix_a"],
        "fix_b": RECIPES["fix_b"]
    }

    for name, preprocess in systems.items():
        print(f"\n--- {name.upper()} ---")

        table, metrics = run_system(
            rows,
            preprocess=preprocess
        )

        assert len(table) == 24

        table.insert(0, "system", name)
        table.insert(1, "split", "validation")

        summary = make_summary(table)
        print(summary.to_string(index=False))
        table.to_csv(
            BASE / f"validation_{name}.csv",
            index=False
        )

        summary.to_csv(
            BASE / f"summary_{name}.csv",
            index=False
        )

        metrics_by_system[name] = metrics
        if (table["error"] != "").all():
            raise RuntimeError(
                f"Every OCR call failed for {name}. "
                "Fix the OCR setup before freezing."
            )
    chosen = min(
        RECIPES,
        key=lambda name: (
            metrics_by_system[name]["corpus_cer"],
            -metrics_by_system[name]["exact_matches"]
        )
    )

    config = {
        "chosen_recipe": chosen,
        "frozen_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "validation_metrics": metrics_by_system
    }
    with CONFIG_FILE.open("x", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\nChosen recipe:", chosen)
    print("Config frozen:", CONFIG_FILE)


if __name__ == "__main__":
    main()
