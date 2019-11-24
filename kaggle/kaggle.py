import os
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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
    # SVM is pretty bad - AVOID
    # learner = SVC(C=1.0, kernel="rbf")

    # Gave 0.83934
    # learner = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=6, min_samples_split=5, bootstrap=True)

    # Gave 0.83934
    # With RobustScaler - 0.83828
    # With MinMaxScaler - 0.81528
    # With StandardScaler - 0.83278
    learner = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=8, min_samples_split=10, bootstrap=True)

    # L2 penalty, Dual=True - 0.81102
    # L2 penalty, Dual=False - 0.79844
    # learner = LogisticRegression(penalty="l2", dual=False, max_iter=1000)

    # learner = KNeighborsClassifier(n_neighbors=10, weights="uniform")


    # With 20 features and RF - 0.82154
    # With 50 features and RF - 0.84745
    # With 50 features and RF, 100 trees, entropy, max depth 8, min samples split 10 - 0.84459
    # With 70 features and RF, 100 trees, gini, max depth 8, min samples split 10 - 0.84848
    # With 70 features and RF, 100 trees, gini, max depth 5, min samples split 10 - 0.84768
    # With 70 features and RF, 100 trees, gini, max depth 8, min samples split 5 - 0.084175
    # KNN 10 neighbors, weights=distance - 0.82638
    # KNN 10 neighbors, weights=uniform - 0.82758
    # KNN 5 neighbors, weights=distance - 0.80565
    pca = PCA(n_components=70)

    # Get data
    x, y = get_train_data()

    # Fit x and get most relevant features
    pca.fit(x)
    x = pca.transform(x)

    # Get test data
    x_test = get_test_data()
    x_test_vals = x_test.loc[:, 1:]

    # Get most relevant features
    x_test_vals = pca.transform(x_test_vals)

    predictions = train_predict(learner, x, y, x_test_vals)
    out = create_output(x_test.index, predictions)

    print(predictions)

    outfn = get_outfile_name()
    print("Writing to {}".format(outfn))
    out.to_csv(outfn, index=None, header=True)
    return

if __name__ == "__main__":
    main()
