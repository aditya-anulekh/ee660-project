import os
import pickle
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)
from sklearn.model_selection import (
    train_test_split
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


def create_ssl_dataset(X, y, keep_labels=0.2):
    # Use sklearn.model_selection.train_test_split to split labeled data
    # Set stratify attribute of train_test_split to perform stratified sampling
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X, y,
        train_size=keep_labels,
        stratify=y
    )
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled


def save_model(model, training_mode='sl', filename=None):
    # Infer filename if not provided
    if filename is None:
        filename = os.path.join(
            config.MODELS_DIR,
            f"{model.__name__}_{training_mode}.pkl"
        )

    # Write the file using pickle
    with open(filename, 'w') as f:
        pickle.dump(model, f)
    return
