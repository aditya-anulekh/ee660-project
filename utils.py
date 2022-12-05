import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)
from sklearn.model_selection import (
    train_test_split
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    f1_score
)
import config


class Preprocessing:
    @staticmethod
    def encode_labels(dataframe: pd.DataFrame, col: str):
        le = LabelEncoder()
        le.fit(dataframe[col])
        dataframe[f'{col}_encoded'] = le.transform(dataframe[col])
        return dataframe

    @staticmethod
    def scale_data(dataframe: pd.DataFrame, cols=None, ss=None):
        if ss is None:
            if cols:
                ss = StandardScaler()
                ss.fit(dataframe[cols])
                dataframe[cols] = ss.transform(dataframe[cols])
            else:
                ss = StandardScaler()
                ss.fit(dataframe)
                dataframe = ss.transform(dataframe)
        else:
            if cols:
                dataframe[cols] = ss.transform(dataframe[cols])
            else:
                dataframe = ss.transform(dataframe)
        return dataframe, ss


def get_data(data_file, columns=None):
    data = pd.read_csv(data_file, names=columns)

    # Convert labels to 0 and 1
    data = Preprocessing.encode_labels(data, 'class')
    data.drop('class', axis=1, inplace=True)
    X, y = data.drop('class_encoded', axis=1), data['class_encoded']

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        random_state=config.RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def get_ssl_data(filepath, columns, missing_label=' unlabeled'):
    dataframe = pd.read_csv(filepath,
                       names=columns,
                       comment="@")

    # Adding space before label because the dataset is formatted like that
    dataframe['class'] = dataframe['class'].replace(missing_label, -1)
    dataframe['class'] = dataframe['class'].replace(' g', 0)
    dataframe['class'] = dataframe['class'].replace(' h', 1)
    X = dataframe.drop('class', axis=1)
    y = dataframe['class']
    return X, y


def create_ssl_dataset(X, y, keep_labels=0.2):
    # Use sklearn.model_selection.train_test_split to split labeled data
    # Set stratify attribute of train_test_split to perform stratified sampling
    X_labeled, X_unlabeled, y_labeled, y_trs = train_test_split(
        X, y,
        train_size=keep_labels,
        stratify=y
    )
    X_train = pd.concat([X_labeled, X_unlabeled]).sort_index()
    y_train = pd.concat([y_labeled, y_trs.replace(y_trs.unique(), -1)]).sort_index()
    return X_train, y_train


def get_model_metrics(model, X_test, y_test):
    metrics = {}
    scores = [
        roc_auc_score,
        accuracy_score,
        f1_score,
        confusion_matrix,
    ]
    for score in scores:
        y_pred = model.predict(X_test)
        metrics.__setitem__(score.__name__, str(score(y_test, y_pred)))
    return metrics


def get_generalization_bound(parameter_grid: list, dataset_sizes: dict):
    bounds = {}
    delta = 0.1
    phases = dataset_sizes.keys()
    for phase in phases:
        if phase == "val":
            M = len(parameter_grid)
        else:
            M = 1
        bound = np.sqrt((1/(2 * dataset_sizes[phase])) * np.log(M / delta))
        bounds.__setitem__(phase, bound)
    return bounds


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data(
        os.path.join(config.DATASET_ROOT,
                     'magic04.data'),
        columns=config.FEATURE_NAMES
    )
    X_ssl, y_ssl = create_ssl_dataset(X_train, y_train)

