import pandas as pd
import numpy as np
from timeit import Timer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import RandomizedPCA


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

    start = True
    for train_i, test_i in skf:
        train_target = [targets[x] for x in train_i]
        train_feats = data[train_i]

        # pca = PCA()
        # pca.fit(train_feats)
        # pca.transform(train_feats)

        clf.fit(train_feats, train_target)

        test_target = [targets[x] for x in test_i]
        test_feats = data[test_i]

        # pca.transform(test_feats)

        pred_targets = clf.predict(test_feats)
        acc = accuracy_score(test_target, pred_targets)
        # print(acc)
        conf_m = confusion_matrix(test_target, pred_targets)
        if start:
            start = False
            cm = conf_m
            acc_list = [acc]
            tot_acc = acc
        else:
            cm += conf_m
            acc_list.append(acc)
            tot_acc += acc

    # print(acc_list)
    print(cm)
    avg_cls_acc = ((cm[0, 0] / (cm[0, 1] + cm[0, 0])) + (cm[1, 0] / (cm[1, 1] + cm[1, 0]))) / 2
    print('average class accuracy: ', avg_cls_acc)
    avg_acc = tot_acc / num_folds
    print('average accuracy across ', num_folds, ' folds: ', avg_acc)

    # print(cross_val_score(clf, data, targets, cv=skf))
    # pred_target = cross_val_predict(clf, data, targets, cv=skf)
    # print(confusion_matrix(targets[skf], pred_target))


def runCls():
    train_data, train_labels = get_data("data\\trainingset.txt")
    test_data, test_labels = get_data("data\\queries.txt")

    # clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
    # clf = GaussianNB()
    # create_model(clf, train_data, train_labels, 20)

    # clf = MLPClassifier(activation='logistic', tol=1e-4,algorithm='adam', warm_start=True, alpha=1e-6, max_iter=500, hidden_layer_sizes=(5, 2), random_state=2, verbose=True)
    # create_model(clf, train_data, train_labels, 10)
    # clf = AdaBoostClassifier(base_estimator=LogisticRegression(tol=1))
    # create_model(clf, train_data, train_labels, 10)
    clf = svm.SVC(kernel='sigmoid', decision_function_shape='ovr')
    create_model(clf, train_data, train_labels, 10)

    # for i in range(1,10):
    #     clf = AdaBoostClassifier(base_estimator=LogisticRegression(tol=i))
    #     create_model(clf, train_data, train_labels, 10)
    #     clf = svm.SVC(kernel='sigmoid', decision_function_shape='ovr')
    #     create_model(clf, train_data, train_labels, 10)
    #     # clf = LogisticRegression(tol=i/100.0)
    #     clf = KNeighborsClassifier(n_neighbors=10 * i, n_jobs=-1, algorithm='brute')
    #     create_model(clf, train_data, train_labels, 10)


def main():
    t = Timer(lambda: runCls())
    print('runtime: ', t.timeit(number=1))


if __name__ == "__main__":
    main()
