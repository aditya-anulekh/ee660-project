import os
import pickle
from utils import (
    get_data,
    get_model_metrics
)
import config


def load_model(filepath, training_mode='sl'):
    if not os.path.isabs(filepath):
        filepath = os.path.join(config.MODELS_DIR,
                                f"{training_mode}_models",
                                filepath)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data(
        os.path.join(config.DATASET_ROOT, 'magic04.data'),
        config.FEATURE_NAMES
    )
    models = os.listdir(os.path.join(config.MODELS_DIR, 'sl_models'))
    for model_path in models:
        model = load_model(model_path)
        train_metrics = get_model_metrics(model, X_train, y_train)
        test_metrics = get_model_metrics(model, X_test, y_test)
        print(f"{model}\n, {train_metrics}, {test_metrics}")
    pass
