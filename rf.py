import random
import numpy as np
from argparse import ArgumentParser
from util import import_file, accuracy
from dt import DecisionTree

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to use for analysis", type=str)
    parser.add_argument("--num-trees", help="Number of decision trees to plant in the forest", type=int, default=20)
    parser.add_argument("--sampling-ratio", help="Percentage of the dataset used to create subsamples", type=float, default=0.5)
    parser.add_argument("--max-depth", help="Max depth of the tree", type=int, default=3)
    parser.add_argument("--min-size", help="Min number of samples required to potentially split a node", type=int, default=2)
    return parser

class RandomForest:
    def __init__(self, num_trees, sampling_ratio, max_depth, min_size):
        self.num_trees = num_trees
        self.sampling_ratio = sampling_ratio
        self.max_depth = max_depth
        self.min_size = min_size
        self.trees = None
        return

    def subsample(self, x, ratio=None):
        if ratio is None:
            ratio = self.sampling_ratio
        samples = []
        num_samples = int(len(x) * ratio)
        while len(samples) < num_samples:
            idx = random.randrange(len(x))
            samples.append(x[idx])
        return samples

    def _predict(self, x):
        num_features = len(x[0]) - 1
        # Predict the labels for x using every tree
        predictions = [tree.predict(x) for tree in self.trees]
        predictions = np.asarray(predictions)
        # Choose the winning prediction for each feature, with a simple majority count
        return [np.bincount(fx).argmax() for fx in np.transpose(predictions)]

    def fit(self, x):
        trees = []
        for _ in range(self.num_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_size=self.min_size)
            subsample = self.subsample(x, self.sampling_ratio)
            tree.fit(subsample)
            trees.append(tree)
        self.trees = trees
        return trees

    def predict(self, x):
        if not self.trees:
            raise ValueError("Random Forest has not been planted yet")
        return self._predict(x)

def main():
    args = setup_argparser().parse_args()

    filename = args.file
    num_trees = args.num_trees
    sampling_ratio = args.sampling_ratio
    max_depth = args.max_depth
    min_size = args.min_size

    x, labels = import_file(filename)
    rf = RandomForest(num_trees=num_trees, sampling_ratio=sampling_ratio, max_depth=max_depth, min_size=min_size)
    rf.fit(x)
    predictions = rf.predict(x)
    print(predictions)
    print(accuracy(labels, predictions))
    return

if __name__=="__main__":
    main()
