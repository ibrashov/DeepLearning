
from pathlib import Path
from PIL import Image
import pytesseract
import easyocr
import numpy as np
import csv

BASE = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE / "midterm_images"
RESULTS_FOLDER = BASE / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

reader = easyocr.Reader(["en"], gpu=False)

extensions = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}

images = sorted(
    p for p in IMAGE_FOLDER.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)

rows = []
detail_rows = []

for image_path in images:
    print(f"\nProcessing: {image_path.name}")

    try:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_array = np.array(rgb_image)

        tess_data = pytesseract.image_to_data(
            rgb_image,
            lang="eng",
            output_type=pytesseract.Output.DICT
        )

        tess_lines = []
        tess_confidences = []

        for i, word in enumerate(tess_data["text"]):
            word = word.strip()
            conf = float(tess_data["conf"][i])

            if word and conf >= 0:
                tess_lines.append(
                    f"{word} | Confidence: {conf:.2f}%"
                )
                tess_confidences.append(conf)

                detail_rows.append({
                    "image": image_path.name,
                    "model": "Tesseract",
                    "text": word,
                    "confidence_percent": round(conf, 2)
                })

        tess_avg = (
            np.mean(tess_confidences)
            if tess_confidences else 0.0
        )

        easy_results = reader.readtext(
            image_array,
            detail=1
        )

        easy_lines = []
        easy_confidences = []

        for bbox, text, conf in easy_results:
            confidence_percent = float(conf) * 100

            easy_lines.append(
                f"{text} | Confidence: "
                f"{confidence_percent:.2f}%"
            )

            easy_confidences.append(confidence_percent)

            detail_rows.append({
                "image": image_path.name,
                "model": "EasyOCR",
                "text": text,
                "confidence_percent": round(
                    confidence_percent, 2
                )
            })

        easy_avg = (
            np.mean(easy_confidences)
            if easy_confidences else 0.0
        )

        # =====================
        # SAVE RESULTS
        # =====================

        output = (
            "TESSERACT:\n"
            + "\n".join(tess_lines)
            + f"\n\nAverage Confidence: {tess_avg:.2f}%"
            + "\n\nEASYOCR:\n"
            + "\n".join(easy_lines)
            + f"\n\nAverage Confidence: {easy_avg:.2f}%"
        )

        output_path = RESULTS_FOLDER / (
            image_path.stem + ".txt"
        )

        output_path.write_text(
            output,
            encoding="utf-8"
        )

        rows.append({
            "image": image_path.name,
            "tesseract_avg_confidence": round(tess_avg, 2),
            "easyocr_avg_confidence": round(easy_avg, 2)
        })

        print(
            f"Tesseract Confidence: {tess_avg:.2f}%"
        )
        print(
            f"EasyOCR Confidence: {easy_avg:.2f}%"
        )

        print(f"Completed: {image_path.name}")

    except Exception as e:
        print(
            f"Error processing {image_path.name}: {e}"
        )

# Save average confidence comparison
with open(
    RESULTS_FOLDER / "comparison.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "tesseract_avg_confidence",
            "easyocr_avg_confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

# Save confidence for every detected text
with open(
    RESULTS_FOLDER / "confidence_details.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "model",
            "text",
            "confidence_percent"
        ]
    )

    writer.writeheader()
    writer.writerows(detail_rows)

print(
    f"\nFinished! Processed "
    f"{len(rows)} of {len(images)} images."
)
