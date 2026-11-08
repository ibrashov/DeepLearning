from pathlib import Path
import pandas as pd
from PIL import Image
from rapidfuzz.distance import Levenshtein
def normalize(text):
    text = text.upper()
    result = ""
    for char in text:
        if not char.isspace():
            result += char
    return result
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
        "corpur_cer": corpur_cer,
        "exact_matches": exact_matchs

    }
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / "week9_homework_data"
CSV_FILE = DATA_DIR/"manifest.csv"
def ocr(image):
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
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
if __name__ == "__main__":
    df = pd.read_csv(CSV_FILE)
    validation = df[df["split"] == "validation"]

    rows = validation.to_dict("records")

    assert len(rows) == 24
    table, metrics = run_system(rows)
    print(table.to_string(index=False))

    print("\nBaseline metrics")
    print("Images:", metrics["images"])
    print("Reference chars:", metrics["reference_chars"])
    print("Total edits:", metrics["total_edits"])

    print(f"Corpus CER: {metrics['corpur_cer']:.2%}")

    print("Exact matches:", metrics["exact_matches"])
    print("OCR errors:", (table["error"] != "").sum())
    print("Empty predictions:",
          (table["prediction"] == "").sum())

    # Save results
    output = BASE / "validation_results.csv"
    table.to_csv(output, index=False)

    print("\nSaved:", output)