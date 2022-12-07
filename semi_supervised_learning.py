"""""
SSL methods
* LabelPropagation
* SelfTraining with different base estimators
* LabelSpreading
* GMM
* S3VM
"""

import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.metrics import roc_auc_score, accuracy_score
from models import (
    GaussianMixtureModel,
    SelfLearner
)
from third_party.qns3vm import QN_S3VM
from utils import (
    get_model_metrics,
    get_data,
    create_ssl_dataset
)
from train import (
    create_pipeline,
    save_model,
)
import config


def train_self_learner():
    X_train, X_test, y_train, y_test = get_data(
        os.path.join(config.DATASET_ROOT,
                     'magic04.data'),
        columns=config.FEATURE_NAMES
    )

    X, y, X_trs, y_trs = create_ssl_dataset(X_train, y_train)

    estimators = [
        AdaBoostClassifier,
        DecisionTreeClassifier,
        KNeighborsClassifier,
        LogisticRegression,
        RandomForestClassifier,
        SVC
    ]

    model_kwargs = {
        'SVC': {
            'kernel': 'linear',
            'probability': True
        }
    }

    # Train self learning model with the above estimators as base
    for estimator in estimators:
        print(f"Training self learning with {estimator.__name__} as base")
        if model_kwargs.get(estimator.__name__) is not None:
            pipeline = create_pipeline(SelfLearner, feature_selection=False,
                                       base_estimator=estimator,
                                       **model_kwargs.get(estimator.__name__))
        else:
            pipeline = create_pipeline(SelfLearner, feature_selection=False,
                                       base_estimator=estimator)

        pipeline.fit(X, y)
        save_model(pipeline, 'ssl', f'SelfTraining_{estimator.__name__}.pkl')
        train_metrics = get_model_metrics(pipeline, X_trs, y_trs)
        test_metrics = get_model_metrics(pipeline, X_test, y_test)
        metric_output = f"{config.MODELS_DIR}/ssl_models/self_{estimator.__name__}.json"
        with open(metric_output, 'w') as f:
            json.dump({
                'train': train_metrics,
                'test': test_metrics
            }, f, indent=4)
        print(f"Train Metrics: {train_metrics}")
        print(f"Test Metrics: {test_metrics}")


def train_label_prop():
    X_train, X_test, y_train, y_test = get_data(
        os.path.join(config.DATASET_ROOT,
                     'magic04.data'),
        columns=config.FEATURE_NAMES
    )

    X, y, X_trs, y_trs = create_ssl_dataset(X_train, y_train)

    pipeline = create_pipeline(LabelPropagation, feature_selection=False,
                               kernel='knn')
    pipeline.fit(X, y)
    save_model(pipeline, 'ssl', f'label_prop.pkl')

    train_metrics = get_model_metrics(pipeline, X_trs, y_trs)
    test_metrics = get_model_metrics(pipeline, X_test, y_test)
    metric_output = f"{config.MODELS_DIR}/ssl_models/label_prop.json"
    with open(metric_output, 'w') as f:
        json.dump({
            'train': train_metrics,
            'test': test_metrics
        }, f, indent=4)


def train_label_spreading():
    X_train, X_test, y_train, y_test = get_data(
        os.path.join(config.DATASET_ROOT,
                     'magic04.data'),
        columns=config.FEATURE_NAMES
    )

    X, y, X_trs, y_trs = create_ssl_dataset(X_train, y_train)

    pipeline = create_pipeline(LabelSpreading, feature_selection=False,
                               kernel='knn')
    pipeline.fit(X, y)
    save_model(pipeline, 'ssl', f'label_spreading.pkl')

    train_metrics = get_model_metrics(pipeline, X_trs, y_trs)
    test_metrics = get_model_metrics(pipeline, X_test, y_test)
    metric_output = f"{config.MODELS_DIR}/ssl_models/label_spreading.json"
    with open(metric_output, 'w') as f:
        json.dump({
            'train': train_metrics,
            'test': test_metrics
        }, f, indent=4)


def train_gmm():
    X_train, X_test, y_train, y_test = get_data(
        os.path.join(config.DATASET_ROOT,
                     'magic04.data'),
        columns=config.FEATURE_NAMES
    )

    X, y, X_trs, y_trs = create_ssl_dataset(X_train, y_train)

    pipeline = create_pipeline(GaussianMixtureModel, feature_selection=False)
    pipeline.fit(X, y)
    save_model(pipeline, 'ssl', f'gmm.pkl')

    train_metrics = get_model_metrics(pipeline, X_trs, y_trs)
    test_metrics = get_model_metrics(pipeline, X_test, y_test)
    print(train_metrics, test_metrics)
    metric_output = f"{config.MODELS_DIR}/ssl_models/gmm.json"
    with open(metric_output, 'w') as f:
        json.dump({
            'train': train_metrics,
            'test': test_metrics
        }, f, indent=4)


def train_ssl_baseline():
    X_train, X_test, y_train, y_test = get_data(
        os.path.join(config.DATASET_ROOT,
                     'magic04.data'),
        columns=config.FEATURE_NAMES
    )

    X, y, X_trs, y_trs = create_ssl_dataset(X_train, y_train)

    pipeline = create_pipeline(LabelPropagation, feature_selection=False,
                               kernel='knn', n_neighbors=1)
    pipeline.fit(X, y)
    save_model(pipeline, 'ssl', f'baseline.pkl')

    train_metrics = get_model_metrics(pipeline, X_trs, y_trs)
    test_metrics = get_model_metrics(pipeline, X_test, y_test)
    metric_output = f"{config.MODELS_DIR}/ssl_models/baseline.json"
    with open(metric_output, 'w') as f:
        json.dump({
            'train': train_metrics,
            'test': test_metrics
        }, f, indent=4)


def train_s3vm():
    # Training code for S3VM is different since it expects labels as
    # {-1, 1} as opposed to {0, 1}
    def get_s3vm_data(filepath, columns, missing_label=' unlabeled'):
        dataframe = pd.read_csv(filepath,
                                names=columns,
                                comment="@")

        # Adding space before label because the dataset is formatted like that
        # dataframe['class'] = dataframe['class'].replace(missing_label, -1)
        dataframe['class'] = dataframe['class'].replace(' g', -1)
        dataframe['class'] = dataframe['class'].replace(' h', 1)
        X = dataframe.drop('class', axis=1)
        y = dataframe['class']
        return X, y

    X, y = get_s3vm_data(
        os.path.join('magic-ssl20-10-1tra.dat'),
        columns=config.FEATURE_NAMES,
    )

    X_test, y_test = get_s3vm_data(
        os.path.join('magic-ssl20-10-1tst.dat'),
        columns=config.FEATURE_NAMES
    )

    X_trs, y_trs = get_s3vm_data(
        os.path.join('magic-ssl20-10-1trs.dat'),
        columns=config.FEATURE_NAMES
    )

    # Create training data
    X_label = X[y != ' unlabeled'].copy()
    y_label = y[y != ' unlabeled'].copy()
    X_unlabeled = X[y == ' unlabeled'].copy()

    s3vm = QN_S3VM(X_label.values.tolist(),
                   y_label.values.tolist(),
                   X_unlabeled.values.tolist(),
                   random.Random())
    # Train and get transduction labels
    y_out = s3vm.train()

    # Predict on testing data
    y_pred = s3vm.getPredictions(X_test.values.tolist())

    # Since the model file is too large we saved the model's predictions
    # to generate results
    with open('s3vm_train.pkl', 'wb') as f:
        pickle.dump({'y_out': y_out, 'y_trs': y_trs}, f)

    with open('s3vm_test.pkl', 'wb') as f:
        pickle.dump({'y_test': y_test, 'y_pred': y_pred}, f)
    return


def compare_with_sl():
    # Since we got the highest accuracies with Random Forest
    # based self learning
    # We tune and train a random forest classifier with just the
    # labeled data

    X_train, X_test, y_train, y_test = get_data(
        os.path.join(config.DATASET_ROOT,
                     'magic04.data'),
        columns=config.FEATURE_NAMES
    )

    sl_train_acc = []
    sl_train_roc = []
    ssl_train_acc = []
    ssl_train_roc = []

    for ratio_labeled in np.arange(0.1, 0.8, 0.1):
        X, y, X_trs, y_trs = create_ssl_dataset(X_train,
                                                y_train,
                                                keep_labels=ratio_labeled)

        # Train SSL model
        ssl_model = create_pipeline(SelfLearner, feature_selection=False,
                                    n_iter=10,
                                    base_estimator=RandomForestClassifier)
        ssl_model.fit(X, y)
        ssl_train_roc.append(roc_auc_score(y_test, ssl_model.predict(X_test)))
        ssl_train_acc.append(accuracy_score(y_test, ssl_model.predict(X_test)))

        # Get only the labeled data
        X_labeled = X[y != -1]
        y_labeled = y[y != -1]

        sl_model = create_pipeline(RandomForestClassifier,
                                   feature_selection=False)
        sl_model.fit(X_labeled, y_labeled)
        sl_train_roc.append(roc_auc_score(y_test,
                                          sl_model.predict(X_test)))
        sl_train_acc.append(accuracy_score(y_test,
                                           sl_model.predict(X_test)))
    plt.plot(np.arange(0.1, 0.8, 0.1), sl_train_acc,
             label="Supervised Learning")
    plt.plot(np.arange(0.1, 0.8, 0.1), ssl_train_acc,
             label="Semi Supervised Learning")
    plt.xlabel("Ratio of Labeled Points")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig(f"{config.PLOTS_DIR}/ssl_results/comparison.png")
    plt.show()
    return


if __name__ == '__main__':
    # train_self_learner()
    # train_label_prop()
    # train_label_spreading()
    # train_gmm()
    # train_ssl_baseline()
    compare_with_sl()
    pass
