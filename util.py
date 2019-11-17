import numpy as np
from sklearn.metrics import precision_score, recall_score

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

# def precision_score(ltrue, lpred, cm=None):
#     assert len(ltrue) == len(lpred), "Unequal number of labels in truth and predictions"
#     if cm is None:
#         cm = confusion_matrix(ltrue, lpred)
#     true_pos = np.diag(cm)
#     false_pos = np.sum(cm, axis=0) - true_pos
#     false_neg = np.sum(cm, axis=1) - true_pos
#     prec = np.sum(true_pos / (true_pos + false_pos))
#     return prec
#
# def recall_score(ltrue, lpred, cm=None):
#     assert len(ltrue) == len(lpred), "Unequal number of labels in truth and predictions"
#     if cm is None:
#         cm = confusion_matrix(ltrue, lpred)
#     true_pos = np.diag(cm)
#     false_pos = np.sum(cm, axis=0) - true_pos
#     false_neg = np.sum(cm, axis=1) - true_pos
#     recall = np.sum(true_pos / (true_pos + false_neg))
#     return recall

def f1_score(ltrue, lpred):
    assert len(ltrue) == len(lpred), "Unequal number of labels in truth and predictions"
    precision = precision_score(ltrue, lpred)
    recall = recall_score(ltrue, lpred)
    return 2 * (precision * recall) / (precision + recall)

def get_metrics(ltrue, lpred):
    """
    Return precision, recall, and F-1 score
    """
    assert len(ltrue) == len(lpred), "Unequal number of labels in truth and predictions"
    return precision_score(ltrue, lpred), recall_score(ltrue, lpred), f1_score(ltrue, lpred)

