# CSE 601 - Project 3 - Classification Algorithms

## How to run

### Decision Tree, and Random Forest

The Decision tree and Random forest algorithms are implemented in Python 3, and require the following packages to be installed:

1. Numpy
2. Scikit-Learn

You can run the algorithms with the following example options:

```python3
python3 dt.py --file ./data/project3_dataset2.txt --max-depth 3 --min-size 2
```

```python3
python3 rf.py --file ./data/project3_dataset2.txt --num-trees 50 --sampling-ratio 0.5 --max-depth 4 --min-size 2 --features-ratio 0.5
```

To find all options you can run with, simply use the `--help` flag with either script.

## TODO:
1. Enforce unique feature splits on decision trees
2. Fill this README with info on how to run stuff:
