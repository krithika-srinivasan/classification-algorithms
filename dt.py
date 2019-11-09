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

class TreeNode:
    def __init__(self, left, right, index, value):
        self.left = left
        self.right = right
        self.index = index
        self.value = value

    def __repr__(self):
        l = r = "ChildNode"
        if not isinstance(self.left, TreeNode):
            l = "Class={0}".format(self.left)
        if not isinstance(self.right, TreeNode):
            r = "Class={0}".format(self.right)
        return "<Node left={0}, right={1}, split index={2}, split value={3}>".format(l, r, self.index, self.value)

    @staticmethod
    def show_tree(root, depth=0):
        SPACER = " "
        print("{0}[X{1} < {2:.4f}]".format(SPACER * depth, root.index, root.value))
        if isinstance(root.left, TreeNode):
            TreeNode.show_tree(root.left, depth + 1)
        else:
            print("{0}[Class {1}]".format(SPACER * (depth + 1), root.left))
        if isinstance(root.right, TreeNode):
            TreeNode.show_tree(root.right, depth + 1)
        else:
            print("{0}[Class {1}]".format(SPACER * (depth + 1), root.right))
        return

class DecisionTree:
    def __init__(self, max_depth=3, min_size=2):
        self.max_depth = max_depth
        self.min_size = min_size
        return

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

    def _split(self, x, index, split_val):
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
                groups = self._split(x, idx, row[idx])
                gini = self.gini_index(groups, uniq_classes)
                if gini < b_score:
                    b_index = idx
                    b_value = row[idx]
                    b_score = gini
                    b_groups = groups
        return b_index, b_value, b_groups

    def to_terminal(self, group):
        klasses = [row[-1] for row in group]
        return max(set(klasses), key=klasses.count)

    def split(self, node, depth=1):
        left, right = node.left, node.right
        if not left or not right:
            node.left = node.right = self.to_terminal(left + right)
            return
        if depth >= self.max_depth:
            node.left, node.right = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= self.min_size:
            node.left = self.to_terminal(left)
        else:
            li, lv, (ll, lr) = self.select_best_split(left)
            node.left = TreeNode(ll, lr, li, lv)
            self.split(node.left, depth + 1)
        if len(right) <= self.min_size:
            node.right = self.to_terminal(right)
        else:
            ri, rv, (rl, rr) = self.select_best_split(right)
            node.right = TreeNode(rl, rr, ri, rv)
            self.split(node.right, depth + 1)
        return

    def fit(self, x):
        root_idx, root_val, (root_left, root_right) = self.select_best_split(x)
        root = TreeNode(root_left, root_right, root_idx, root_val)
        self.split(root)
        return root



def main():
    args = setup_argparser().parse_args()
    filename = args.file
    x = import_file(filename)
    dt = DecisionTree(max_depth=3, min_size=2)
    tree = dt.fit(x)
    TreeNode.show_tree(tree)
    return

if __name__ == "__main__":
    main()
