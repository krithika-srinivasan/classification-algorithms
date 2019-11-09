def import_file(path):
    result = []
    labels = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            vals = line.split("\t")
            row = []
            for idx, val in enumerate(vals):
                val = val.strip()
                if not val:
                    continue
                try:
                    if idx == len(vals) - 1:
                        row.append(int(val))
                        labels.append(int(val))
                    else:
                        row.append(float(val))
                except:
                    row.append(val)
            result.append(row)
    assert all([len(row) == len(result[0]) for row in result]), "Not all rows have the same number of values"
    return result, labels

def accuracy(ltrue, lpred):
    assert len(ltrue) == len(lpred), "Unequal number of labels in truth and predictions"
    count = 0
    for t, p in zip(ltrue, lpred):
        if t == p:
            count += 1
    return count / len(ltrue)

