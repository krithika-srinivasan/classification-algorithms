import sys
from argparse import ArgumentParser

INT_MAX = sys.maxsize
INT_MIN = -(INT_MAX) - 1

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to use for analysis", type=str)
    return parser

def import_file(path):
    result = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            vals = line.split("\t")
            row = []
            for val in vals:
                val = val.strip()
                if not val:
                    continue
                try:
                    row.append(float(val))
                except:
                    row.append(val)
            result.append(row)
    assert all([len(row) == len(result[0]) for row in result]), "Not all rows have the same number of values"
    return result

class DecisionTree:
    def __init__(self):
        pass

    def gini_index(self, groups, classes):
        num_values = sum([len(group) for group in groups])
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            for klass in classes:
                classes_in_group = [row[-1] for row in group]
                p = classes_in_group.count(klass) / size
                score += p * p
            gini += (1.0 - score) * (size / num_values)
        return gini

    def split(self, x, index, split_val):
        left, right = [], []
        if isinstance(split_val, (float, int)):
            for row in x:
                if row[index] < split_val:
                    left.append(row)
                else:
                    right.append(row)
        elif isinstance(split_val, str):
            for row in x:
                if row[index] == split_val:
                    left.append(row)
                else:
                    right.append(row)
        return left, right

    def select_best_split(self, x):
        # Assuming the last value in a row is its class label
        uniq_classes = list(set([row[-1] for row in x]))
        num_features = len(x[0]) - 1
        b_index, b_value, b_score, b_groups = INT_MAX, INT_MAX, INT_MAX, None
        for idx in range(num_features):
            for row in x:
                groups = self.split(x, idx, row[idx])
                gini = self.gini_index(groups, uniq_classes)
                if gini < b_score:
                    b_index = idx
                    b_value = row[idx]
                    b_score = gini
                    b_groups = groups
        return b_index, b_value, b_groups

def main():
    args = setup_argparser().parse_args()
    filename = args.file
    x = import_file(filename)
    dt = DecisionTree()
    idx, value, (left, right) = dt.select_best_split(x)
    print(idx, value, len(left), len(right), len(x))
    return

if __name__ == "__main__":
    main()
