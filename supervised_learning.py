"""
Supervised Learning

**Baselines**
* Trivial Classifier
* Logistic Regression

**Without Over/Under Sampling**
* RandomForestClassifier
* MLPClassifier
* KNeighborsClassifier
* DecisionTreeClassifier
* SVC
* AdaBoostClassifier

**Sampling Techniques**
* Above classifiers are implemented in a pipeline with each of the
below-mentioned techniques
* Oversampling
    * SMOTE
    * ADASYN
* Undersampling
    * RandomUnderSampling
    * EditedNearestNeighbours
* Combination
    * SMOTEENN
"""

import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import (
    RandomUnderSampler,
    EditedNearestNeighbours
)
from imblearn.combine import SMOTEENN
from models import TrivialClassifier
from train import grid_search, save_model
from utils import get_data, get_generalization_bound
from grid_search_params import grid_params
import config


# In[3]:


X_train, X_test, y_train, y_test = get_data(
    os.path.join(config.DATASET_ROOT,
                 'magic04.data'),
    columns=config.FEATURE_NAMES
)

# Trivial Classifier
trivial_classifier = TrivialClassifier()
trivial_classifier.fit(X_train, y_train)
save_model(trivial_classifier, filename='trivial_classifier.pkl')

# Non-trivial Classifier - 1
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
save_model(logistic_regression, filename='logistic_regression.pkl')

# Non-trivial Classifier - 2
one_nn = KNeighborsClassifier(n_neighbors=1)
one_nn.fit(X_train, y_train)
save_model(one_nn, filename='one_nn.pkl')


estimators = [
    RandomForestClassifier,
    MLPClassifier,
    KNeighborsClassifier,
    DecisionTreeClassifier,
    SVC,
    AdaBoostClassifier,
]

# No sampling
for estimator in estimators:
    grid_search(estimator, X_train, y_train, save=True)

# SMOTE
for estimator in estimators:
    grid_search(estimator,
                X_train, y_train,
                save=True, sampler=SMOTE)

# ADASYN
for estimator in estimators:
    grid_search(estimator,
                X_train, y_train,
                save=True, sampler=ADASYN)

# RandomUnderSampler
for estimator in estimators:
    grid_search(estimator,
                X_train, y_train,
                save=True, sampler=RandomUnderSampler)

# EditedNearestNeighbours
for estimator in estimators:
    grid_search(estimator,
                X_train, y_train,
                save=True, sampler=EditedNearestNeighbours)

# SMOTEENN
for estimator in estimators:
    grid_search(estimator,
                X_train, y_train,
                save=True, sampler=SMOTEENN)


# Get Generalization Bounds
for estimator in estimators:
    bound = get_generalization_bound(
        parameter_grid=list(ParameterGrid(grid_params[estimator.__name__])),
        dataset_sizes={
            'val': len(X_train) // config.N_FOLDS,
            'test': len(X_test)
        }
    )
    print(f"{estimator.__name__}, {bound}")
