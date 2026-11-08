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
if __name__ == "__main__":
    refs = ["BOOK", "CAT", "R7"]
    preds = ["BOK", "CATS", "R7"]
    result = score(refs, preds)
    print("Images:", result["images"])
    print("Reference chars:", result["reference_chars"])
    print("Total edits:", result["total_edits"])
    print(f"Corpus CER: {result['corpur_cer']:.2%}")
    print("Exact matches:", result["exact_matches"])

    assert result["images"] == 3
    assert result["reference_chars"] == 9
    assert result["total_edits"] == 2
    assert result["exact_matches"] == 1

    print("All score tests passed!")