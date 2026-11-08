
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from PIL import Image
from rapidfuzz.distance import Levenshtein

if __package__:
    from .part3_3 import DATA_DIR, fix_a, fix_b, normalize, ocr, score
else:
    from part3_3 import DATA_DIR, fix_a, fix_b, normalize, ocr, score
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
CSV_FILE = DATA_DIR/"manifest.csv"
RECIPES ={
    "fix_a": fix_a,
    "fix_b": fix_b
}
BASE = Path(__file__).resolve().parent
CONFIG_FILE = BASE / "chosen_config.json"
PREDICTIONS_FILE = BASE / "predictions.csv"
TEST_MARKER = BASE / "test_started.txt"


def main():
    # 1. Test is forbidden before freeze
    if not CONFIG_FILE.exists():
        raise RuntimeError(
            "Freeze the configuration before testing!"
        )

    # Prevent running test a second time
    if TEST_MARKER.exists() or PREDICTIONS_FILE.exists():
        raise RuntimeError(
            "Test has already been started or completed."
        )

    # 2. Read the frozen recipe
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = json.load(f)

    chosen = config["chosen_recipe"]
    assert chosen in RECIPES

    print("Frozen recipe:", chosen)
    print("Frozen at:", config["frozen_at_utc"])

    # 3. Load the 72 validation predictions
    validation_tables = []

    for name in ["baseline", "fix_a", "fix_b"]:
        path = BASE / f"validation_{name}.csv"
        table = pd.read_csv(path, keep_default_na=False)

        assert len(table) == 24
        assert (table["system"] == name).all()
        assert (table["split"] == "validation").all()

        validation_tables.append(table)

    # 4. Load 24 test images
    df = pd.read_csv(CSV_FILE)
    test_rows = df[
        df["split"] == "test"
    ].to_dict("records")

    assert len(test_rows) == 24

    # Record that test evaluation has started
    TEST_MARKER.write_text(
        datetime.now(timezone.utc).isoformat(),
        encoding="utf-8"
    )

    test_tables = []

    # 5. Test baseline and frozen recipe
    for name, preprocess in [
        ("baseline", None),
        (chosen, RECIPES[chosen])
    ]:
        print(f"\nTEST: {name}")

        table, metrics = run_system(
            test_rows,
            preprocess=preprocess
        )

        assert len(table) == 24

        table.insert(0, "system", name)
        table.insert(1, "split", "test")

        print(make_summary(table).to_string(index=False))

        print(f"Corpus CER: {metrics['corpus_cer']:.2%}")
        print("Exact matches:", metrics["exact_matches"])
        print("OCR errors:", (table["error"] != "").sum())

        table.to_csv(
            BASE / f"test_{name}.csv",
            index=False
        )

        test_tables.append(table)

    # 6. Combine all 120 predictions
    predictions = pd.concat(
        validation_tables + test_tables,
        ignore_index=True
    )

    assert len(predictions) == 120

    assert (
        predictions["split"] == "validation"
    ).sum() == 72

    assert (
        predictions["split"] == "test"
    ).sum() == 48

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    print("\nSaved:", PREDICTIONS_FILE)
    print("Total predictions:", len(predictions))


if __name__ == "__main__":
    main()
