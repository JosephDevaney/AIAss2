import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer


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


def main():
    data, labels = get_data("data\\trainingset.txt")


if __name__ == "__main__":
    main()
