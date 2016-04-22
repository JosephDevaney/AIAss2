import pandas as pd
import numpy as np
from timeit import Timer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import VotingClassifier

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import os


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
    data_ids = data["id"]
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

    return feat_data, target_labels, data_ids


def create_model(clf, data, targets, num_folds):
    skf = StratifiedKFold(targets, n_folds=num_folds)

    start = True
    for k, (train_i, test_i) in enumerate(skf):
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
        print("Accuracy for fold " + str(k) + " is: ", )
        print(str(acc) + "\n")
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

    avg_acc = tot_acc / num_folds

    print('average accuracy across ', num_folds, ' folds: ', avg_acc)

    # print(cross_val_score(clf, data, targets, cv=skf))
    # pred_target = cross_val_predict(clf, data, targets, cv=skf)
    # print(confusion_matrix(targets[skf], pred_target))


def predict_queries(clf, train_feats, train_target, queries):
    clf.fit(train_feats, train_target)
    pred_labels = clf.predict(queries)

    return pred_labels


def write_preds(query_ids, pred_labels):
    contain_sol_dir = False
    for d in os.listdir():
        if d == "solutions":
            contain_sol_dir = True

    if not contain_sol_dir:
        os.mkdir("./solutions")

    filename = "./solutions/C12731135+C12425568.txt"
    solutions = open(filename, 'w')

    [solutions.write(qid + "," + pred_l + "\n") for qid, pred_l in zip(query_ids, pred_labels)]
    solutions.close()


def create_even_dataset(data, labels, ids):
    data_a = [i for i, label in enumerate(labels) if label == "TypeA"]
    data_b = [i for i, label in enumerate(labels) if label == "TypeB"]

    short_data = slice_array(data, data_a, data_b)
    short_labels = slice_lists(labels, data_a, data_b)
    short_ids = slice_lists(ids, data_a, data_b)

    return short_data, short_labels, short_ids


def slice_array(data, slice_a, slice_b):
    short_data = data[slice_a[:len(slice_b)]]
    short_data = np.vstack((short_data, data[slice_b]))

    return short_data


def slice_lists(data, slice_a, slice_b):
    short_data = data[slice_a[:len(slice_b)]].tolist()
    short_data.extend(data[slice_b])

    return short_data


def run_cls():
    train_data, train_labels, train_ids = get_data("data\\trainingset.txt")
    test_data, test_labels, test_ids = get_data("data\\queries.txt")

    train_data, train_labels, train_ids = create_even_dataset(train_data, train_labels, train_ids)

    # clf1 = svm.SVC(class_weight='balanced', kernel='poly', decision_function_shape='ovr')
    # clf2 = LogisticRegression(class_weight='balanced', solver='sag', max_iter=1000)
    # clf3 = svm.SVC(kernel='sigmoid', decision_function_shape='ovr', class_weight='balanced')
    #
    # vclf = VotingClassifier(estimators=[('lsvc', clf1), ('lr', clf2), ('sigsvc', clf3)], voting='soft')
    # create_model(vclf, train_data, train_labels, 10)

    # clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
    # clf = GaussianNB()
    # create_model(clf, train_data, train_labels, 20)

    # clf = MLPClassifier(activation='logistic', tol=1e-4, algorithm='adam', warm_start=True, alpha=1e-6, max_iter=500,
    # hidden_layer_sizes=(5, 2), random_state=2, verbose=True)
    # create_model(clf, train_data, train_labels, 10)
    # clf = AdaBoostClassifier(base_estimator=LogisticRegression(tol=1))
    # create_model(clf, train_data, train_labels, 10)
    # clf = svm.SVC(kernel='rbf', class_weight={'TypeB': 1.24})
    # create_model(clf, train_data, train_labels, 10)
    clf = LogisticRegressionCV(cv=5, class_weight='balanced', n_jobs=-1, solver='sag', max_iter=1000)
    #
    # create_model(clf, train_data, train_labels, 5)

    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                      'class_weight': [{'TypeB': w} for w in [1.25, 1.26, 1.27, 1.28, 1.29]],
    #                      'C': [1, 10, 100, 1000]}]
    #
    # scores = ['precision', 'recall']
    #
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
    #                        scoring='%s_weighted' % score)
    #     clf.fit(train_data, train_labels)
    #
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     print()
    #     for params, mean_score, scores in clf.grid_scores_:
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean_score, scores.std() * 2, params))
    #     print()
    #
    #     print("Detailed classification report:")
    #     print()
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     print()
    #     y_true, y_pred = train_labels, clf.predict(train_data)
    #     print(classification_report(y_true, y_pred))
    #     print()
    #
    #     # create_model(clf, train_data, train_labels, 10, tuned_parameters)
    #     #     clf = AdaBoostClassifier(base_estimator=LogisticRegression(tol=i))
    #     #     create_model(clf, train_data, train_labels, 10)
    #     #     clf = svm.SVC(kernel='sigmoid', decision_function_shape='ovr')
    #     #     create_model(clf, train_data, train_labels, 10)
    #     #     clf = KNeighborsClassifier(n_neighbors=10 * i, n_jobs=-1, algorithm='brute')
    #     #     create_model(clf, train_data, train_labels, 10)
    # for i in range(1,10):
    # clf = BaggingClassifier(LogisticRegression(class_weight='balanced'), max_samples=0.1, max_features=0.1)
    # create_model(clf, train_data, train_labels, 10)

    # clf1 = BaggingClassifier(LogisticRegression(class_weight='balanced'), max_samples=0.1, max_features=0.1)
    # clf2 = RandomForestClassifier(random_state=1, class_weight='balanced')
    # clf3 = KNeighborsClassifier(n_neighbors=10, n_jobs=-1, algorithm='brute')
    #
    # clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

    
    #     clf = AdaBoostClassifier(base_estimator=LogisticRegression(tol=i))
    #     create_model(clf, train_data, train_labels, 10)
    #     clf = svm.SVC(kernel='sigmoid', decision_function_shape='ovr')
    #     create_model(clf, train_data, train_labels, 10)
    #     clf = KNeighborsClassifier(n_neighbors=10 * i, n_jobs=-1, algorithm='brute')
    #     create_model(clf, train_data, train_labels, 10)

    create_model(clf, train_data, train_labels, 5)
    results = predict_queries(clf, train_data, train_labels, test_data)
    write_preds(test_ids, results)


def main():
    t = Timer(lambda: run_cls())
    print('runtime: ', t.timeit(number=1))


if __name__ == "__main__":
    main()
