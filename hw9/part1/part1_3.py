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

def wrong_collapse(path, blank="ø"):
    without_blank =[]
    for token in path:
        if token!=blank:
            without_blank.append(token)
    merged =[]
    previous = None
    for token in without_blank:
        if token!=previous:
            merged.append(token)
        previous = token
    return "".join(merged)
path = ["B", "B", "ø", "O", "O", "ø","O", "K", "K"]
path1 = ["B", "B", "ø", "O", "O", "ø", "O", "K", "K"]
path2 = ["ø", "B", "ø", "O", "ø", "O", "K", "ø"]
result1 = ctc_collapse(path1)
result2 = ctc_collapse(path2)
print("Path 1:", result1)
print("Path 2:", result2)
assert result1 == "BOOK"
assert result2 == "BOOK"
assert path1 != path2

print("Both paths decode to BOOK")
