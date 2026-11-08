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
print("Correct:", ctc_collapse(path))
print("Wrong:  ", wrong_collapse(path))