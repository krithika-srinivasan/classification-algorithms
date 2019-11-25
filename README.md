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

### Naive Bayesian, and Nearest Neighbours

Both these algorithms are written in R, and require the following packages:

1. tidyVerse
2. matrixStats
3. magrittr
4. reshape2

You can run each of the algorithms using the following command:

```bash
Rscript naive_bayesian.R
```
OR
```bash
Rscript nearest_neighbour.R
```

To configure parameters, you will have to edit the files themselves.
