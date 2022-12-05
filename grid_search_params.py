"""
Models tested in the project
* LogisticRegression
* DecisionTree
* RandomForest
* SVC
* MLP
* KNN
"""

import numpy as np

# Name of the estimator object defined in the pipeline
pipeline_estimator = 'estimator__'
pipeline_skb = 'skb__'

grid_params = {
    'DecisionTreeClassifier': {
        f'{pipeline_skb}k': range(8, 11),
        f'{pipeline_estimator}criterion': ["gini", "entropy", "log_loss"],
        f'{pipeline_estimator}min_samples_split': range(10, 30, 10)
    },
    'RandomForestClassifier': {
        f'{pipeline_skb}k': range(8, 11),
        f'{pipeline_estimator}n_estimators': range(80, 110, 10),
        f'{pipeline_estimator}criterion': ["gini", "entropy", "log_loss"],
        f'{pipeline_estimator}min_samples_split': range(10, 30, 10)
    },
    'SVC': {
        f'{pipeline_skb}k': range(8, 11),
        f'{pipeline_estimator}C': np.arange(0.1, 1, 0.1),
        f'{pipeline_estimator}kernel': ['rbf', 'poly', 'linear']

    },
    'MLPClassifier': {
        f'{pipeline_skb}k': range(8, 11),
        f'{pipeline_estimator}hidden_layer_sizes': [(100,), (200, 100)]
    },
    'KNeighborsClassifier': {
        f'{pipeline_skb}k': range(8, 11),
        f'{pipeline_estimator}n_neighbors': range(15, 20, 2)
    },
    'AdaBoostClassifier': {
        f'{pipeline_skb}k': range(8, 11),
        f'{pipeline_estimator}n_estimators': range(80, 110, 10)
    }
}
