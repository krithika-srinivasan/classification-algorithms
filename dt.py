import sys
import random
from argparse import ArgumentParser
from cv import CrossValidation
from util import import_file, accuracy, get_metrics

INT_MAX = sys.maxsize
INT_MIN = -(INT_MAX) - 1

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to use for analysis", type=str)
    parser.add_argument("--max-depth", help="Max depth of the tree", type=int, default=3)
    parser.add_argument("--min-size", help="Min number of samples required to potentially split a node", type=int, default=2)
    return parser

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
        print("{0}[X{1} < {2}]".format(SPACER * depth, root.index, root.value))
        if isinstance(root.left, TreeNode):
            TreeNode.show_tree(root.left, depth + 1)
        else:
            print("{0}[Class {1}]".format(SPACER * (depth + 1), root.left))
        if isinstance(root.right, TreeNode):
            TreeNode.show_tree(root.right, depth + 1)
        else:
            print("{0}[Class {1}]".format(SPACER * (depth + 1), root.right))
        return

    @staticmethod
    def verify_tree(root):
        l, r = False, False
        if isinstance(root.left, TreeNode):
            l = TreeNode.verify_tree(root.left)
        else:
            l = isinstance(root.left, int)
        if isinstance(root.right, TreeNode):
            r = TreeNode.verify_tree(root.right)
        else:
            r = isinstance(root.right, int)
        return l and r


class DecisionTree:
    """
    Binary decision tree.
    """
    def __init__(self, max_depth=3, min_size=2, features_ratio=1.0):
        """
        max_depth - Max depth of the tree
        min_size - Minimum number of elements that must be present in a node, to consider a split
        features_ratio - Percentage of features to select while selecting a split
        """
        self.max_depth = max_depth
        self.min_size = min_size
        self.features_ratio = features_ratio
        self.root = None
        self._num_features = 0
        self.features_used = set()
        return

    def gini_index(self, groups, classes):
        num_values = sum([len(group) for group in groups])
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            classes_in_group = [row[-1] for row in group]
            for klass in classes:
                p = classes_in_group.count(klass) / size
                score += p * p
            gini += (1.0 - score) * (size / num_values)
        return gini

    def _split(self, x, index, split_val):
        """
        Split the feature set into two branches, using the given
        split value.
        """
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
        """
        Choose the best feature value to split the dataset on.
        """
        # Assuming the last value in a row is its class label
        uniq_classes = list(set([row[-1] for row in x]))
        num_features = self._num_features
        num_features_selected = int(num_features * self.features_ratio)
        selected_features_indexes = random.sample(range(num_features), num_features_selected)
        b_index, b_value, b_score, b_groups = INT_MAX, INT_MAX, INT_MAX, None
        for idx in selected_features_indexes:
            if idx in self.features_used:
                continue
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
        """
        Return majority label for the group.
        """
        klasses = [row[-1] for row in group]
        return max(set(klasses), key=klasses.count)

    def check_if_same_terminal(self, left, right):
        return self.to_terminal(left) == self.to_terminal(right)

    def split(self, node, depth=1):
        """
        Recursively split the node, until we reach no-split condition.
        """
        left, right = node.left, node.right
        if not left or not right:
            node.left = node.right = self.to_terminal(left + right)
            return

        if depth >= self.max_depth or len(self.features_used) == self._num_features:
            node.left, node.right = self.to_terminal(left), self.to_terminal(right)
            return

        if len(left) <= self.min_size:
            node.left = self.to_terminal(left)
        else:
            fidx, fvalue, ls = self.select_best_split(left)
            if ls is None:
                node.left = self.to_terminal(left)
            else:
                ll, lr = ls
                self.features_used.add(fidx)
                node.left = TreeNode(ll, lr, fidx, fvalue)
                self.split(node.left, depth + 1)

        if len(right) <= self.min_size:
            node.right = self.to_terminal(right)
        else:
            fidx, fvalue, rs = self.select_best_split(right)
            if rs is None:
                node.right = self.to_terminal(right)
            else:
                rl, rr = rs
                self.features_used.add(fidx)
                node.right = TreeNode(rl, rr, fidx, fvalue)
                self.split(node.right, depth + 1)
        return

    def _predict(self, node, target):
        """
        Traverse the node, until we reach a leaf, giving us a class value.
        """
        idx, val = node.index, node.value
        tval = target[idx]
        comparator = None
        if isinstance(tval, str):
            comparator = lambda x, y: x == y
        elif isinstance(tval, (float, int)):
            comparator = lambda x, y: x < y
        if comparator(tval, val):
            if isinstance(node.left, TreeNode):
                return self._predict(node.left, target)
            else:
                return node.left
        else:
            if isinstance(node.right, TreeNode):
                return self._predict(node.right, target)
            else:
                return node.right
        return None

    def fit(self, x):
        self.features_used = set()
        self._num_features = len(x[0]) - 1
        root_idx, root_val, (root_left, root_right) = self.select_best_split(x)
        self.features_used.add(root_idx)
        root = TreeNode(root_left, root_right, root_idx, root_val)
        self.split(root)
        # assert TreeNode.verify_tree(root), "Leaf nodes are not integers"
        self.root = root
        return root

    def predict(self, x):
        if not self.root:
            raise ValueError("Decision Tree has not been fitted")
        return [self._predict(self.root, row) for row in x]

def main():
    args = setup_argparser().parse_args()

    filename = args.file
    max_depth = args.max_depth
    min_size = args.min_size

    x, labels = import_file(filename)
    dt = DecisionTree(max_depth=max_depth, min_size=min_size)
    tree = dt.fit(x)
    TreeNode.show_tree(tree)
    predicted_labels = dt.predict(x)
    p, r, f1 = get_metrics(labels, predicted_labels, class_label=1)
    acc = accuracy(labels, predicted_labels)
    print("Naive results")
    print("Accuracy: {}, Precision: {}, Recall: {}, F-1: {}".format(acc, p, r, f1))

    ten_cv = CrossValidation(k=10)
    dt = DecisionTree(max_depth=max_depth, min_size=min_size)
    train_scores, val_scores, *_ = ten_cv.cross_validate(dt, x, labels)
    print("10-fold cross validation")
    print("Training scores: {0}\nValidation scores: {1}".format(train_scores, val_scores))
    return

if __name__ == "__main__":
    main()
