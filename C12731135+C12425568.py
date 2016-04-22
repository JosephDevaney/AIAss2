# Joseph Devaney - C12731135
# Darren Britton - C12425568

import os

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# get_data takes a filename and path to a file containing the data
# The contents of the file are read into a DataFrame using the pandas read_csv function
# The categorical and continuous features are separated
# IDs and target labels are removed from the categorical features
# Categorical features are transformed into vectors using one-hot encoding
# Features are combined into one object
#
# Returns a DataFrame of the new features, the target labels and the IDs
def get_data(file):
    col_names = ["id", "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                 "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]

    data = pd.read_csv(file, header=None, names=col_names, index_col=False)
    cont_features = []
    cat_features = []

    for col in col_names:
        # If the type of the column is object it is a string. Else it is a numerical type
        if data[col].dtype == object:
            cat_features.append(col)
        else:
            cont_features.append(col)

    # Extract the labels and IDs before removing these from the feature set
    target_labels = data["y"]
    data_ids = data["id"]
    cat_features.remove("id")
    cat_features.remove("y")

    cont_data = data[cont_features].astype("int64")

    # Create the categorical grouping by dropping the continuous features
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
            # Use numpy hstack function to add a new column to the matrix of data. It is a horizontal stack
            cat_data_trans = np.hstack((cat_data_trans, vec_cat_data))

    # Combine categorical and continuous features
    feat_data = np.hstack((cont_data.as_matrix(), vec_cat_data))

    return feat_data, target_labels, data_ids


# This function, create_model, is used to test a model for accuracy using the training dataset
# It takes a classifier object (must implement fit() and predict()), the data to be used, labels and the
# number of folds to use.
# A Stratified Cross-Fold Validation is used to ensure an equal split of classes is taken in each train/test split
# During each 'fold' the data is split into the training and testing sets.
# The classifier is fit() with training data and predicted on testing data
# The accuracy score and confusion matrix is calculated for each fold and aggregated
# At the end the average accuracy is calculated and shown with the total Confusion Matrix
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

        # Extract a list of the corretc labels for this test set
        test_target = [targets[x] for x in test_i]
        test_feats = data[test_i]

        # pca.transform(test_feats)

        pred_targets = clf.predict(test_feats)
        acc = accuracy_score(test_target, pred_targets)
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

    print(cm)

    avg_acc = tot_acc / num_folds

    print('average accuracy across ', num_folds, ' folds: ', avg_acc)


# Used to fit() and predict() a classifier on the full testing set of queries
def predict_queries(clf, train_feats, train_target, queries):
    clf.fit(train_feats, train_target)
    pred_labels = clf.predict(queries)

    return pred_labels


# Takes the test dataset IDs and predicted labels and writes these to the correct file.
def write_preds(query_ids, pred_labels):
    contain_sol_dir = False
    # Check if the solutions folder exists and create it if it doesn't
    for d in os.listdir():
        if d == "solutions":
            contain_sol_dir = True

    if not contain_sol_dir:
        os.mkdir("./solutions")

    # Open the file
    filename = "./solutions/C12731135+C12425568.txt"
    solutions = open(filename, 'w')

    # Write the ID,Label on a new line for every pair in query_id, pred_label
    [solutions.write(qid + "," + pred_l + "\n") for qid, pred_l in zip(query_ids, pred_labels)]
    solutions.close()


# Used in testing to see if noticeable changes in results were found when using an even number of
# instances for each class
def create_even_dataset(data, labels, ids):
    data_a = [i for i, label in enumerate(labels) if label == "TypeA"]
    data_b = [i for i, label in enumerate(labels) if label == "TypeB"]

    short_data = slice_array(data, data_a, data_b)
    short_labels = slice_lists(labels, data_a, data_b)
    short_ids = slice_lists(ids, data_a, data_b)

    return short_data, short_labels, short_ids


# Used in testing to see if noticeable changes in results were found when using an even number of
# instances for each class
def slice_array(data, slice_a, slice_b):
    short_data = data[slice_a[:len(slice_b)]]
    short_data = np.vstack((short_data, data[slice_b]))

    return short_data


# Used in testing to see if noticeable changes in results were found when using an even number of
# instances for each class
def slice_lists(data, slice_a, slice_b):
    short_data = data[slice_a[:len(slice_b)]].tolist()
    short_data.extend(data[slice_b])

    return short_data


# This function gathers the training and testing datasets.
# It creates the classifier and calls the testing, the predicting and the write to file functions
def main():
    train_data, train_labels, train_ids = get_data("data\\trainingset.txt")
    test_data, test_labels, test_ids = get_data("data\\queries.txt")

    clf = LogisticRegression(n_jobs=-1, class_weight=None)

    create_model(clf, train_data, train_labels, 10)
    results = predict_queries(clf, train_data, train_labels, test_data)
    write_preds(test_ids, results)


if __name__ == "__main__":
    main()
