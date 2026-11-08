def ctc_collapse(path, blank="ø"):
    merged = []
    previous = None

    for token in path:
        if token != previous:
            merged.append(token)
        previous = token
    result = []
    for token in merged:
        if token != blank:
            result.append(token)
    return "".join(result)
assert ctc_collapse([]) == ""
assert ctc_collapse(["ø", "ø"]) == ""
assert ctc_collapse(["A", "A", "A"]) == "A"
assert ctc_collapse(["A", "ø", "A"]) == "AA"
assert ctc_collapse(["C", "C", "ø", "A", "A", "T", "T"]) == "CAT"
print("All CTC tests passed!")
