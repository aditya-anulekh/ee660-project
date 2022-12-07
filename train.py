import os
import warnings
import pickle
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imb_pipeline
from grid_search_params import grid_params
import config


def dummy(*args, **kwargs):
    pass


warnings.warn = dummy


def create_pipeline(estimator, sampler=None, feature_selection=True, **kwargs):
    pipeline_list = []

    # Add objects to the pipeline
    if sampler is not None:
        pipeline_list.append(('sampler', sampler()))

    if feature_selection:
        pipeline_list.extend([
            ('skb', SelectKBest(score_func=mutual_info_classif)),
            ('scaler', MinMaxScaler()),
            ('estimator', estimator(**kwargs))
        ])
    else:
        pipeline_list.extend([
            ('scaler', MinMaxScaler()),
            ('estimator', estimator(**kwargs))
        ])

    # Create the pipeline
    if sampler is not None:
        pipeline = imb_pipeline(pipeline_list)
    else:
        pipeline = Pipeline(pipeline_list)
    return pipeline


def grid_search(model, X_train, y_train, sampler=None, save=False):
    # Create the pipeline with the model and sampler
    pipeline = create_pipeline(model, sampler)

    # Define gridsearch based on the parameters in the config file
    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params.get(model.__name__),
        refit=True,
        cv=config.N_FOLDS,
        scoring='roc_auc',
        verbose=4,
        n_jobs=-1
    )

    # Fit grid search on the training data
    clf.fit(X_train, y_train)

    output_file = 'grid_search_output.txt'

    # Pickle the model
    if save:
        save_model(clf.best_estimator_,
                   filename=f"{model.__name__}_{sampler.__name__ if sampler is not None else None}.pkl")
        with open(output_file, 'a') as f:
            f.write(f"{model.__name__}_{sampler.__name__ if sampler is not None else None}\n")
            f.write(str(clf.best_params_))
            f.write('\n')

    return clf.best_estimator_, clf.best_params_


def save_model(model, training_mode='sl', filename=None):
    # Infer filename if not provided
    if filename is None:
        filename = os.path.join(
            config.MODELS_DIR,
            f"{type(model['estimator']).__name__}_{training_mode}.pkl"
        )

    # Create the filename with appropriate directory structure
    filename = os.path.join(config.MODELS_DIR,
                            f'{training_mode}_models',
                            filename)

    # Write the model file using pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return


if __name__ == "__main__":
    pass
