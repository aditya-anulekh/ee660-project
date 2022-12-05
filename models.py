import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, get_scorer
from scipy.stats import multivariate_normal
import config


class TrivialClassifier(BaseEstimator):
    def __init__(self):
        self.weights = None
        pass

    def fit(self, X, y):
        self.weights = y.value_counts().to_dict()
        return self

    def predict(self, X):
        num_points = X.shape[0]
        return random.choices(list(self.weights.keys()),
                              list(self.weights.values()),
                              k=num_points)


class GaussianMixtureModel(BaseEstimator):
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.cluster_means = None
        self.cluster_cov = None
        self.weights = None
        self.proba = None
        self.n_clusters = None
        self.fitted = False

    def fit(self, X_train:pd.DataFrame, y_train: pd.Series):

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)

        # Separate labeled and unlabeled points
        X_l = X_train[y_train != -1].copy()
        y_l = y_train[y_train != -1].copy()
        X_u = X_train[y_train == -1].copy()
        cluster_labels = y_l.unique()
        self.n_clusters = len(cluster_labels)
        n_iter = 0

        # Concatenate X_l and y_l to get labeled data
        D_l = pd.concat([X_l, y_l], axis=1)

        # Calculate initial means, covariances and cluster weights
        self.cluster_means = D_l.groupby(by=D_l.columns[-1]).mean() \
            .sort_index().to_numpy().reshape((self.n_clusters,
                                              X_train.shape[1],
                                              1))

        self.proba = np.eye(self.n_clusters)[y_l]

        # Initialize cluster covariance
        self.cluster_cov = np.zeros((self.n_clusters,
                                     X_train.shape[1],
                                     X_train.shape[1]))

        # Compute cluster covariance
        for i in range(self.n_clusters):
            dist_mean = X_train - self.cluster_means[i].reshape(-1)
            self.cluster_cov[i, :, :] = np.dot(dist_mean.T, dist_mean)
            self.cluster_cov[i] /= np.sum(self.proba[:, i])

        self.weights = y_l.value_counts(normalize=True).sort_index().to_numpy()

        while n_iter < self.max_iter:
            # Set probabilities of labeled points for every iteration
            self.proba = np.eye(self.n_clusters)[y_l]

            # Expectation Step
            _denom = 0
            unlabeled_proba = np.zeros((X_u.shape[0], self.n_clusters))
            for i in range(self.n_clusters):
                gaussian_pdf = multivariate_normal(
                    mean=self.cluster_means[i].reshape(-1),
                    cov=self.cluster_cov[i])
                _denom += self.weights[i] * gaussian_pdf.pdf(X_u)
            for i in range(self.n_clusters):
                gaussian_pdf = multivariate_normal(
                    mean=self.cluster_means[i].reshape(-1),
                    cov=self.cluster_cov[i])
                unlabeled_proba[:, i] = (self.weights[i] * gaussian_pdf.pdf(X_u))/_denom

            self.proba = np.append(self.proba,
                                   unlabeled_proba, axis=0)

            # Maximization Step
            # Compute the mean
            for i in range(self.n_clusters):
                self.cluster_means[i, :, :] = np.average(
                    X_train,
                    weights=self.proba[:, i],
                    axis=0)[..., np.newaxis]

                dist_mean = X_train - self.cluster_means[i].reshape(-1)
                self.cluster_cov[i, :, :] = np.dot(
                    dist_mean.T,
                    np.multiply(dist_mean, self.proba[:, i].reshape(-1, 1))
                )
                self.cluster_cov[i] /= np.sum(self.proba[:, i])
                self.weights[i] = np.sum(self.proba[:, i])/X_train.shape[0]

            # Increment iterations
            n_iter += 1
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise Exception("Model not fitted! "
                            "Call .fit() before calling predict")

        proba = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            gaussian_pdf = multivariate_normal(
                mean=self.cluster_means[i].reshape(-1),
                cov=self.cluster_cov[i]
            )
            proba[:, i] = gaussian_pdf.pdf(X)
        return np.argmax(proba, axis=1)

    def score(self, X, y, metric='roc_auc'):
        if not self.fitted:
            raise Exception("Model not fitted! "
                            "Call .fit() before calling predict")

        y_pred = self.predict(X)
        scorer = get_scorer(metric)
        return scorer._score_func(y, y_pred)


class SelfLearner(BaseEstimator):
    def __init__(self,
                 base_estimator=LogisticRegression,
                 threshold=0.75,
                 n_iter=100,
                 **kwargs):
        self.base_estimator = base_estimator(**kwargs)
        self.threshold = threshold
        self.n_iter = n_iter
        self.fitted = False
        pass

    def fit(self, X_train, y_train, **kwargs):

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)

        iter_count = 0

        X_label = X_train[y_train != -1].copy()
        y_label = y_train[y_train != -1].copy()
        X_unlabeled = X_train[y_train == -1].copy()

        # Define progress bar
        pbar = tqdm(self.n_iter)

        while len(X_unlabeled) > 0:
            self.base_estimator.fit(X_label, y_label)
            pred_prob = self.base_estimator.predict_proba(X_unlabeled)
            pred = self.base_estimator.predict(X_unlabeled)
            drop_idx = []
            for i in range(pred_prob.shape[1]):
                high_prob_idx = np.where(pred_prob[:, i] > self.threshold)[0]
                drop_idx.extend(list(high_prob_idx))

                # Add newly labelled data to X_label
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

            X_label.reset_index(drop=True, inplace=True)
            X_unlabeled.reset_index(drop=True, inplace=True)
            y_label.reset_index(drop=True, inplace=True)

            iter_count += 1
            pbar.update(1)
            if iter_count > self.n_iter:
                break
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise Exception("Model not fitted! "
                            "Call .fit() before calling predict")
        else:
            return self.base_estimator.predict(X)

    def score(self, X, y):
        if not self.fitted:
            raise Exception("Model not fitted! "
                            "Call .fit() before calling score")
        else:
            return self.base_estimator.score(X, y)


if __name__ == '__main__':

    # Testing GMM with make_blobs dummy data
    X, y, centers = make_blobs(100, n_features=3, centers=2, return_centers=True)
    print(np.cov(X[y == 0], rowvar=False))
    print(np.cov(X[y == 1], rowvar=False))
    X_l, X_u, y_l, y_u = train_test_split(X, y, test_size=0.5, stratify=y)
    y_u = np.ones_like(y_u)*-1
    X_train = np.concatenate([X_l, X_u])
    y_train = np.concatenate([y_l, y_u])
    X_train = pd.DataFrame(X_train, columns=['feature_1', 'feature_2', 'feature_3'])
    y_train = pd.Series(y_train)
    gmm = GaussianMixtureModel(max_iter=1000)
    gmm.fit(X_train, y_train)
    print(gmm.cluster_means, gmm.cluster_cov)
