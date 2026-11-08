
from pathlib import Path
from collections import Counter
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_FOLDER = BASE.parent / "week9_homework_data"
CSV_FILE = DATA_FOLDER / "manifest.csv"
IMAGE_FOLDER = DATA_FOLDER / "images"

rows = pd.read_csv(CSV_FILE)
rows.columns = rows.columns.str.strip()

records = rows.to_dict("records")

print("Total rows:", len(records))
print("Columns:", rows.columns.tolist())
print(rows.head())

# Step 3: Check total rows
assert len(records) == 48, "Expected 48 rows"

print("Total rows check passed!")

# Step 4: Check splits
assert "split" in rows.columns, "Missing split column"

split_counts = Counter(rows["split"])

print("Split counts:", split_counts)

assert len(split_counts) == 2
assert sorted(split_counts.values()) == [24, 24]

print("Split check passed!")

# Step 5: Check base_id
assert "base_id" in rows.columns, "Missing base_id column"

base_splits = {}

for row in records:
    base_id = row["base_id"]
    split = row["split"]

    if base_id in base_splits:
        assert base_splits[base_id] == split, (
            f"Data leakage: {base_id}")
    else:
        base_splits[base_id] = split

print("Base ID check passed!")

# Step 6: Check six base IDs per split
for split in split_counts:
    unique_ids = {
        row["base_id"]
        for row in records
        if row["split"] == split
    }

    print(split, "unique base IDs:", len(unique_ids))
    assert len(unique_ids) == 6

print("Base ID counts passed!")

# Step 7: Check image files
possible_columns = [
    "image_path", "filename", "file_name",
    "path", "image", "file"
]

image_column = next(
    (name for name in possible_columns if name in rows.columns),
    None
)

if image_column is None:
    raise ValueError(
        f"Cannot find image path column. Available: {rows.columns.tolist()}"
    )

for row in records:
    filename = Path(str(row[image_column]))

    if filename.is_absolute():
        image_path = filename
    else:
        candidates = [
            DATA_FOLDER / filename,
            IMAGE_FOLDER / filename,
            BASE.parent / filename
        ]
        image_path = next(
            (path for path in candidates if path.is_file()),
            candidates[0]
        )

    assert image_path.is_file(), (
        f"Missing image: {image_path}"
    )

def normalize(text):
    text = text.upper()
    result = ""
    for char in text:
        if not char.isspace():
            result += char
    return result

assert normalize("Hello World") == "HELLOWORLD"
assert normalize(" wb-4817 ") == "WB-4817"
assert normalize("BOOK-220") == "BOOK-220"
assert normalize("A\nB\tC") == "ABC"
assert normalize("B0OK") == "B0OK"

print("All normalization tests passed!")