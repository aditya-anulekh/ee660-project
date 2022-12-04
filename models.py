import os
import random
from sklearn.base import BaseEstimator
from utils import get_data, get_model_metrics
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


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data(
        os.path.join(config.DATASET_ROOT,
                     'magic04.data'),
        columns=config.FEATURE_NAMES
    )
    model = TrivialClassifier()
    model.fit(X_train, y_train)
    print(get_model_metrics(model, X_test, y_test))
