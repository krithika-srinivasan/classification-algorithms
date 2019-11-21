import numpy as np

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

def has_func(module, func_name):
    return hasattr(module, func_name) and callable(getattr(module, func_name, None))

def confusion_matrix(ltrue, lpred):
    assert len(ltrue) == len(lpred), "Unequal number of labels in truth and predictions"
    num_uniq = len(set(ltrue))
    cm = np.zeros((num_uniq, num_uniq))
    for idx in range(len(ltrue)):
        cm[ltrue[idx], lpred[idx]] += 1
    return cm

def precision_score(ltrue, lpred, cm=None):
    assert len(ltrue) == len(lpred), "Unequal number of labels in truth and predictions"
    if cm is None:
        cm = confusion_matrix(ltrue, lpred)
    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    prec = np.nan_to_num(true_pos / (true_pos + false_pos))
    return prec

def recall_score(ltrue, lpred, cm=None):
    assert len(ltrue) == len(lpred), "Unequal number of labels in truth and predictions"
    if cm is None:
        cm = confusion_matrix(ltrue, lpred)
    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    recall = np.nan_to_num(true_pos / (true_pos + false_neg))
    return recall

def get_metrics(ltrue, lpred, class_label=None):
    """
    Return precision, recall, and F-1 score.
    class_label = Return the metrics for the given label.
    """
    assert len(ltrue) == len(lpred), "Unequal number of labels in truth and predictions"
    cm = confusion_matrix(ltrue, lpred)
    precision = precision_score(ltrue, lpred, cm)
    recall = recall_score(ltrue, lpred, cm)
    f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    if class_label is not None:
        return precision[class_label], recall[class_label], f1[class_label]
    return precision, recall, f1

