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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from models import (
    GaussianMixtureModel,
    SelfLearner
)
from utils import (
    get_ssl_data,
    get_model_metrics
)
from train import create_pipeline, save_model
import config


def train_self_learner():
    X, y = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1tra.dat'),
        columns=config.FEATURE_NAMES,
    )

    X_test, y_test = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1tst.dat'),
        columns=config.FEATURE_NAMES
    )

    X_trs, y_trs = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1trs.dat'),
        columns=config.FEATURE_NAMES
    )

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
        pipeline = create_pipeline(SelfLearner, feature_selection=False,
                                   base_estimator=estimator,
                                   **model_kwargs.get(estimator.__name__))
        pipeline.fit(X, y)
        save_model(pipeline, 'ssl', f'SelfTraining_{estimator.__name__}')
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
    X, y = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1tra.dat'),
        columns=config.FEATURE_NAMES,
    )

    X_test, y_test = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1tst.dat'),
        columns=config.FEATURE_NAMES
    )

    X_trs, y_trs = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1trs.dat'),
        columns=config.FEATURE_NAMES
    )

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
    X, y = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1tra.dat'),
        columns=config.FEATURE_NAMES,
    )

    X_test, y_test = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1tst.dat'),
        columns=config.FEATURE_NAMES
    )

    X_trs, y_trs = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1trs.dat'),
        columns=config.FEATURE_NAMES
    )

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
    X, y = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1tra.dat'),
        columns=config.FEATURE_NAMES,
    )

    X_test, y_test = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1tst.dat'),
        columns=config.FEATURE_NAMES
    )

    X_trs, y_trs = get_ssl_data(
        os.path.join(config.DATASET_ROOT, 'magic-ssl20', 'magic-ssl20-10-1trs.dat'),
        columns=config.FEATURE_NAMES
    )

    pipeline = create_pipeline(GaussianMixtureModel, feature_selection=False)
    pipeline.fit(X, y)
    save_model(pipeline, 'ssl', f'gmm.pkl')

    train_metrics = get_model_metrics(pipeline, X_trs, y_trs)
    test_metrics = get_model_metrics(pipeline, X_test, y_test)
    metric_output = f"{config.MODELS_DIR}/ssl_models/gmm.json"
    with open(metric_output, 'w') as f:
        json.dump({
            'train': train_metrics,
            'test': test_metrics
        }, f, indent=4)


def train_s3vm():
    pass


if __name__ == '__main__':
    # train_self_learner()
    # train_label_prop()
    # train_label_spreading()
    # train_gmm()
    pass
