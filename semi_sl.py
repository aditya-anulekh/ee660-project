import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def dummy(*args, **kwargs):
    pass


warnings.warn = dummy


class SelfLearner(BaseEstimator):
    def __init__(self,
                 base_estimator=LogisticRegression,
                 threshold=0.99,
                 n_iter=1000,
                 **kwargs):
        self.base_estimator = base_estimator(**kwargs)
        self.threshold = threshold
        self.n_iter = n_iter
        pass

    def fit(self, X_label, y_label,
            X_unlabeled, y_unlabeled=None,
            **kwargs):
        iter_count = 0
        # X_label, y_label = X[y != missing_label], y[y != missing_label]
        # X_unlabeled, y_unlabeled = X[y == missing_label], y[y == missing_label]

        while len(X_unlabeled) > 0:
            print(f"{iter_count=}")
            self.base_estimator.fit(X_label, y_label)
            pred_prob = self.base_estimator.predict_proba(X_unlabeled)
            pred = self.base_estimator.predict(X_unlabeled)
            drop_idx = []
            for i in range(pred_prob.shape[1]):
                high_prob_idx = np.where(pred_prob[:, i] > self.threshold)[0]
                drop_idx.extend(list(high_prob_idx))
                # Add newly labelled data to X_label
                # X_label = pd.concat([
                #     X_label,
                #     X_unlabeled.iloc[high_prob_idx]
                # ], axis=0)
                X_label = pd.concat([
                    X_label,
                    X_unlabeled[pred_prob[:, i] > self.threshold]
                ], axis=0)
                y_label = pd.concat([
                    y_label,
                    pd.Series(pred[pred_prob[:, i] > self.threshold])
                ])
            # Remove data from X_unlabeled
            X_unlabeled.drop(
                X_unlabeled.iloc[drop_idx].index,
                inplace=True)

            print(f"{len(X_label)=}")
            print(f"{len(X_unlabeled)=}")

            X_label.reset_index(drop=True, inplace=True)
            X_unlabeled.reset_index(drop=True, inplace=True)
            y_label.reset_index(drop=True, inplace=True)

            plt.scatter(X_label.iloc[:, 0][y_label == 0],
                        X_label.iloc[:, 1][y_label == 0])
            plt.scatter(X_label.iloc[:, 0][y_label == 1],
                        X_label.iloc[:, 1][y_label == 1])
            plt.show()

            iter_count += 1
            if iter_count > self.n_iter:
                break

        return self

    def predict(self):
        pass

    def score(self, X, y):
        pass


if __name__ == "__main__":
    FEATURE_NAMES = [
        'fLength',
        'fWidth',
        'fSize',
        'fConc',
        'fConc1',
        'fAsym',
        'fM3Long',
        'fM3Trans',
        'fAlpha',
        'fDist',
        'class',
    ]


    def encode_labels(dataframe: pd.DataFrame, col):
        le = LabelEncoder()
        le.fit(dataframe[col])
        dataframe[f'{col}_encoded'] = le.transform(dataframe[col])
        return dataframe


    # data = pd.read_csv('data/magic-ssl20/magic-ssl20-10-1tra.dat',
    #                    comment='@', names=FEATURE_NAMES)
    # data = encode_labels(data, 'class')
    # data.drop('class', axis=1, inplace=True)
    # X, y = data.drop('class_encoded', axis=1), data['class_encoded']

    D_l = pd.read_csv('test_labeled.csv')
    D_u = pd.read_csv('test_unlabeled.csv')

    X, y = D_l.loc[:, D_l.columns != '2'], D_l['2']
    model = SelfLearner(n_iter=2, threshold=0.95)
    model.fit(X, y, D_u)
    # print(model.score(X, y))
    # test_func(X, y)
