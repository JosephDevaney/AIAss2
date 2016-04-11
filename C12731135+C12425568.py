import pandas as pd
import numpy as np
from timeit import Timer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import accuracy_score, confusion_matrix


def get_data(file):

    col_names = ["id", "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                 "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]

    data = pd.read_csv(file, header=None, names=col_names, index_col=False)
    cont_features = []
    cat_features = []

    for col in col_names:
        if data[col].dtype == object:
            cat_features.append(col)
        else:
            cont_features.append(col)

    target_labels = data["y"]
    cat_features.remove("id")
    cat_features.remove("y")

    cont_data = data[cont_features].astype("int64")

    cat_data = data.drop(cont_features + ["y"], axis=1)
    start = True
    for cat in cat_features:
        cat_data_temp = cat_data[cat].to_frame(cat)
        cat_data_temp = cat_data_temp.T.to_dict().values()
        vectorizer = DictVectorizer(sparse=False)
        vec_cat_data = vectorizer.fit_transform(cat_data_temp)

        if start:
            cat_data_trans = vec_cat_data
            start = False
        else:
            cat_data_trans = np.hstack((cat_data_trans, vec_cat_data))

    feat_data = np.hstack((cont_data.as_matrix(), vec_cat_data))

    return feat_data, target_labels


def create_model(clf, data, targets, num_folds):
    skf = StratifiedKFold(targets, n_folds=num_folds)

    # for train_i, test_i in skf:
    #     train_target = [targets[x] for x in train_i]
    #     train_feats = data[train_i]
    #
    #     clf.fit(train_feats, train_target)
    #
    #     test_target = [targets[x] for x in test_i]
    #     test_feats = data[test_i]
    #
    #     pred_targets = clf.predict(test_feats)

    print(cross_val_score(clf, data, targets, cv=skf))


def runCls():
    train_data, train_labels = get_data("data\\trainingset.txt")
    test_data, test_labels = get_data("data\\queries.txt")

    for i in range(1,10):
        #clf = svm.SVC(kernel='sigmoid', decision_function_shape='ovr')
        clf = LogisticRegression(tol=i/10.0)
        create_model(clf, train_data, train_labels, 10)


def main():
    t = Timer(lambda: runCls())
    print(t.timeit(number=1))

if __name__ == "__main__":
    main()
