import random
import numpy as np
from argparse import ArgumentParser
from dt import DecisionTree
from cv import CrossValidation
from util import import_file, accuracy

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to use for analysis", type=str)
    parser.add_argument("--num-trees", help="Number of decision trees to plant in the forest", type=int, default=20)
    parser.add_argument("--sampling-ratio", help="Percentage of the dataset used to create subsamples", type=float, default=0.5)
    parser.add_argument("--max-depth", help="Max depth of the tree", type=int, default=3)
    parser.add_argument("--min-size", help="Min number of samples required to potentially split a node", type=int, default=2)
    parser.add_argument("--features-ratio", help="Percentage of features to select while selecting a split", type=float, default=1.0)
    return parser

class RandomForest:
    def __init__(self, num_trees, sampling_ratio, max_depth, min_size, features_ratio):
        """
        num_trees - Number of trees to plant in the forest
        sampling_ratio - Percentage of samples to select to fit to a tree
        max_depth - Max depth of the tree
        min_size - Minimum number of elements that must be present in a node, to consider a split
        features_ratio - Percentage of features to select while selecting a split
        """
        self.num_trees = num_trees
        self.sampling_ratio = sampling_ratio
        self.max_depth = max_depth
        self.min_size = min_size
        self.features_ratio = features_ratio
        self.trees = None
        return

    def subsample(self, x, ratio=None):
        if ratio is None:
            ratio = self.sampling_ratio
        num_samples = int(len(x) * ratio)
        samples = random.sample(x, num_samples)
        return samples

    def _predict(self, x):
        # Predict the labels for x using every tree
        predictions = [tree.predict(x) for tree in self.trees]
        # Choose the winning prediction for each feature, with a simple majority count
        return [np.bincount(feature).argmax() for feature in np.transpose(predictions)]

    def fit(self, x):
        trees = []
        for _ in range(self.num_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_size=self.min_size, features_ratio=self.features_ratio)
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
    features_ratio = args.features_ratio

    x, labels = import_file(filename)
    rf = RandomForest(num_trees=num_trees, sampling_ratio=sampling_ratio, max_depth=max_depth, min_size=min_size, features_ratio=features_ratio)
    rf.fit(x)
    predictions = rf.predict(x)
    print("Naive accuracy", accuracy(labels, predictions))

    ten_cv = CrossValidation(k=10)
    rf = RandomForest(num_trees=num_trees, sampling_ratio=sampling_ratio, max_depth=max_depth, min_size=min_size, features_ratio=features_ratio)
    train_scores, val_scores = ten_cv.cross_validate(rf, x, labels)
    print("Training scores: {0}, validation scores: {1}".format(train_scores, val_scores))
    return

if __name__=="__main__":
    main()
