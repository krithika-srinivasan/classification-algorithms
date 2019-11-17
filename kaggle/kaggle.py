import os
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

BASE_LOC = "./data"
TRAIN = "{0}/train_features.csv".format(BASE_LOC)
TRAIN_LABELS = "{0}/train_label.csv".format(BASE_LOC)
TEST = "{0}/test_features.csv".format(BASE_LOC)
OUTPUT_LOC = "./output"

def get_train_data():
    x = pd.read_csv(TRAIN, header=None, index_col=0)
    y = pd.read_csv(TRAIN_LABELS, header=0, index_col=0)
    return x.loc[:, 1:], y.loc[:, "label"]

def get_test_data():
    x = pd.read_csv(TEST, header=None, index_col=0)
    return x

def create_output(ids, predictions):
    df = pd.DataFrame(columns=["id", "label"])
    df["id"] = ids
    df["label"] = predictions
    return df

def train_predict(learner, x, y, x_test):
    learner.fit(x, y)
    return learner.predict(x_test)

def get_outfile_name():
    return "{0}/submission_{1}.csv".format(OUTPUT_LOC, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def main():
    if not os.path.isdir(OUTPUT_LOC):
        os.makedirs(OUTPUT_LOC, exist_ok=True)

    # Gave 0.83006
    # learner = RandomForestClassifier(n_estimators=50, criterion="gini", max_depth=5, min_samples_split=2, bootstrap=True)

    # Gave 0.80757 - bad settings?
    # learner = SVC(C=1.0, kernel="rbf")

    # Gave 0.83934
    # learner = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=6, min_samples_split=5, bootstrap=True)

    # Gave 0.83934
    learner = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=6, min_samples_split=8, bootstrap=True)

    x, y = get_train_data()
    x_test = get_test_data()
    x_test_vals = x_test.loc[:, 1:]
    predictions = train_predict(learner, x, y, x_test_vals)
    out = create_output(x_test.index, predictions)
    print(predictions)
    print(out)
    outfn = get_outfile_name()
    print("Writing to {}".format(outfn))
    out.to_csv(outfn, index=None, header=True)
    return

if __name__ == "__main__":
    main()
