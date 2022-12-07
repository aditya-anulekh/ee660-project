import os
import sys
from glob import glob
import argparse
import pickle
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


samplers = [
    'None',
    'SMOTE',
    'ADASYN',
    'RandomUnderSampler',
    'EditedNearestNeighbours',
    'SMOTEENN'
]


X_train, X_test, y_train, y_test = get_data(
    os.path.join(config.DATASET_ROOT, 'magic04.data'),
    config.FEATURE_NAMES
)


def generate_sl_results():
    roc_fig, _ = plt.subplots(2, 3)
    roc_fig.set_size_inches(9, 6)
    acc_fig, _ = plt.subplots(2, 3)
    acc_fig.set_size_inches(9, 6)

    for i, sampler in enumerate(samplers):
        print(sampler)
        model_paths = glob(f'{config.MODELS_DIR}/sl_models/*{sampler}.pkl')

        roc_scores = {'train': [], 'test': []}
        accuracies = {'train': [], 'test': []}
        models = []

        # Add trivial and baseline models to all graphs
        baseline_models = ['trivial_classifier.pkl',
                           'one_nn.pkl',
                           'logistic_regression.pkl']

        for model_path in baseline_models:
            model = load_model(model_path)
            models.append(type(model).__name__)
            print(f"\t{type(model).__name__}")

            train_metrics = get_model_metrics(model, X_train, y_train)
            test_metrics = get_model_metrics(model, X_test, y_test)

            # Append train scores
            roc_scores['train'].append(float(train_metrics['roc_auc_score']))
            accuracies['train'].append(float(train_metrics['accuracy_score']))

            # Append test scores
            roc_scores['test'].append(float(test_metrics['roc_auc_score']))
            accuracies['test'].append(float(test_metrics['accuracy_score']))

        for model_path in model_paths:
            model = load_model(model_path)
            models.append(type(model['estimator']).__name__)
            print(f"\t{type(model['estimator']).__name__}")

            train_metrics = get_model_metrics(model, X_train, y_train)
            test_metrics = get_model_metrics(model, X_test, y_test)

            # Append train scores
            roc_scores['train'].append(float(train_metrics['roc_auc_score']))
            accuracies['train'].append(float(train_metrics['accuracy_score']))

            # Append test scores
            roc_scores['test'].append(float(test_metrics['roc_auc_score']))
            accuracies['test'].append(float(test_metrics['accuracy_score']))

        roc_fig.axes[i].plot(roc_scores['train'], label="ROC AUC - Train",
                             color='r')
        roc_fig.axes[i].plot(roc_scores['test'], label="ROC AUC - Test",
                             color='g')
        acc_fig.axes[i].plot(accuracies['train'], label="Accuracy - Train",
                             color='b')
        acc_fig.axes[i].plot(accuracies['test'], label="Accuracy - Test",
                             color='c')
        roc_fig.axes[i].set_title(f"Sampler: {sampler}")
        acc_fig.axes[i].set_title(f"Sampler: {sampler}")

        if i > 2:
            roc_fig.axes[i].set_xticks(range(len(models)))
            roc_fig.axes[i].set_xticklabels(models, rotation=90)
            acc_fig.axes[i].set_xticks(range(len(models)))
            acc_fig.axes[i].set_xticklabels(models, rotation=90)
        else:
            roc_fig.axes[i].set_xticks([])
            acc_fig.axes[i].set_xticks([])

    roc_lines = [
        Line2D([0], [0], color='r'),
        Line2D([0], [0], color='g'),
    ]
    acc_lines = [
        Line2D([0], [0], color='b'),
        Line2D([0], [0], color='c'),
    ]

    roc_fig.legend(roc_lines, ["ROC AUC - Train",
                               "ROC AUC - Test"])

    acc_fig.legend(acc_lines, ["Accuracy - Train",
                               "Accuracy - Test"])

    roc_fig.suptitle("ROC AUV Scores for various sampling methods")
    acc_fig.suptitle("Accuracy Scores for various sampling methods")

    roc_fig.tight_layout()
    acc_fig.tight_layout()
    # roc_fig.savefig(f'{config.PLOTS_DIR}/sl_results/roc_curves.png')
    # acc_fig.savefig(f'{config.PLOTS_DIR}/sl_results/acc_curves.png')
    plt.show()


def generate_sl_tables():
    with open("tables.txt", "w") as f:
        for i, sampler in enumerate(samplers):
            print("\\begin{table}[h]", file=f)
            print("\\caption{%s}" % sampler, file=f)
            print("\\centering", file=f)
            print("\\begin{tabular}{| c | c | c | c | c |}", file=f)
            print("\\hline", file=f)
            print(
                "Model & ROC Train & Accuracy Train & ROC Test & Accuracy Test \\\\",
                file=f)
            print("\\hline", file=f)

            model_paths = glob(f'{config.MODELS_DIR}/sl_models/*{sampler}.pkl')

            # Add trivial and baseline models to all graphs
            baseline_models = ['trivial_classifier.pkl',
                               'one_nn.pkl',
                               'logistic_regression.pkl']

            if sampler == "None":
                for model_path in baseline_models:
                    model = load_model(model_path)

                    train_metrics = get_model_metrics(model, X_train, y_train)
                    test_metrics = get_model_metrics(model, X_test, y_test)

                    print(f"{type(model).__name__} & "
                          f"{round(float(train_metrics['roc_auc_score']), 3)} & "
                          f"{round(float(train_metrics['accuracy_score']), 3)} & "
                          f"{round(float(test_metrics['roc_auc_score']), 3)} & "
                          f"{round(float(test_metrics['accuracy_score']), 3)} \\\\",
                          file=f)

            for model_path in model_paths:
                model = load_model(model_path)

                train_metrics = get_model_metrics(model, X_train, y_train)
                test_metrics = get_model_metrics(model, X_test, y_test)

                print(f"{type(model['estimator']).__name__} & "
                      f"{round(float(train_metrics['roc_auc_score']), 3)} & "
                      f"{round(float(train_metrics['accuracy_score']), 3)} & "
                      f"{round(float(test_metrics['roc_auc_score']), 3)} & "
                      f"{round(float(test_metrics['accuracy_score']), 3)} \\\\",
                      file=f)
                print("\\hline", file=f)
            print("\\end{tabular}", file=f)
            print("\\end{table}", file=f)


def generate_ssl_results():
    # Semi-Supervised Learning Results

    roc_fig, _ = plt.subplots(1, 1)
    roc_fig.set_size_inches(6, 6)
    acc_fig, _ = plt.subplots(1, 1)
    acc_fig.set_size_inches(6, 6)

    model_paths = glob(f'{config.MODELS_DIR}/ssl_models/*.pkl')

    roc_scores = {'train': [], 'test': []}
    accuracies = {'train': [], 'test': []}
    models = ['S3VM']

    # Handle S3VM separately
    with open(f'{config.MODELS_DIR}/ssl_models/s3vm_train.pkl1', 'rb') as f:
        out = pickle.load(f)
        accuracies['train'].append(accuracy_score(out['y_out'], out['y_trs']))
        roc_scores['train'].append(roc_auc_score(out['y_out'], out['y_trs']))

    with open(f'{config.MODELS_DIR}/ssl_models/s3vm_test.pkl1', 'rb') as f:
        out = pickle.load(f)
        accuracies['test'].append(accuracy_score(out['y_test'], out['y_pred']))
        roc_scores['test'].append(roc_auc_score(out['y_test'], out['y_pred']))

    for model_path in model_paths:
        print(model_path)
        model = load_model(model_path, 'ssl')
        train_metrics = get_model_metrics(model, X_train, y_train)
        test_metrics = get_model_metrics(model, X_test, y_test)
        if type(model['estimator']).__name__ == "SelfLearner":
            models.append(
                f"Self_{type(model['estimator'].base_estimator).__name__}")
        else:
            models.append(type(model['estimator']).__name__)

        # Append train scores
        roc_scores['train'].append(float(train_metrics['roc_auc_score']))
        accuracies['train'].append(float(train_metrics['accuracy_score']))

        # Append test scores
        roc_scores['test'].append(float(test_metrics['roc_auc_score']))
        accuracies['test'].append(float(test_metrics['accuracy_score']))

    # Change baseline model name
    models[1] = 'LabelPropagation (k=1)'

    roc_order = np.argsort(roc_scores['train'])
    # acc_order = np.argsort(accuracies['train'])
    models = np.array(models)

    roc_fig.axes[0].plot(np.array(roc_scores['train'])[roc_order], color='r')
    roc_fig.axes[0].plot(np.array(roc_scores['test'])[roc_order], color='g')
    acc_fig.axes[0].plot(np.array(accuracies['train'])[roc_order], color='b')
    acc_fig.axes[0].plot(np.array(accuracies['test'])[roc_order], color='c')

    roc_fig.axes[0].set_xticks(range(len(models)))
    roc_fig.axes[0].set_xticklabels(np.array(models)[roc_order], rotation=90)
    acc_fig.axes[0].set_xticks(range(len(models)))
    acc_fig.axes[0].set_xticklabels(np.array(models)[roc_order], rotation=90)

    roc_lines = [
        Line2D([0], [0], color='r'),
        Line2D([0], [0], color='g'),
    ]
    acc_lines = [
        Line2D([0], [0], color='b'),
        Line2D([0], [0], color='c'),
    ]

    roc_fig.legend(roc_lines, ["ROC AUC - Train",
                               "ROC AUC - Test"])

    acc_fig.legend(acc_lines, ["Accuracy - Train",
                               "Accuracy - Test"])

    roc_fig.suptitle("ROC AUV Scores")
    acc_fig.suptitle("Accuracy Scores")

    roc_fig.tight_layout()
    acc_fig.tight_layout()
    roc_fig.savefig(f'{config.PLOTS_DIR}/ssl_results/roc_curves.png')
    acc_fig.savefig(f'{config.PLOTS_DIR}/ssl_results/acc_curves.png')
    plt.show()


def generate_ssl_tables():
    with open("ssl_tables.txt", "w") as f:
        print("\\begin{table}[h]", file=f)
        print("\\caption{%s}" % "SSL Model Results", file=f)
        print("\\centering", file=f)
        print("\\begin{tabular}{| c | c | c | c | c |}", file=f)
        print("\\hline", file=f)
        print(
            "Model & ROC Train & Accuracy Train & ROC Test & Accuracy Test \\\\",
            file=f)
        print("\\hline", file=f)

        model_paths = glob(f'{config.MODELS_DIR}/ssl_models/*.pkl')

        for model_path in model_paths:
            model = load_model(model_path)

            if type(model['estimator']).__name__ == "SelfLearner":
                model_name = f"Self_{type(model['estimator'].base_estimator).__name__}"
            else:
                model_name = type(model['estimator']).__name__

            train_metrics = get_model_metrics(model, X_train, y_train)
            test_metrics = get_model_metrics(model, X_test, y_test)

            print(f"{model_name} & "
                  f"{round(float(train_metrics['roc_auc_score']), 3)} & "
                  f"{round(float(train_metrics['accuracy_score']), 3)} & "
                  f"{round(float(test_metrics['roc_auc_score']), 3)} & "
                  f"{round(float(test_metrics['accuracy_score']), 3)} \\\\",
                  file=f)
            print("\\hline", file=f)
        print("\\end{tabular}", file=f)
        print("\\end{table}", file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument to generate only final results
    parser.add_argument('--final_only', type=bool, default=True)

    # Add argument to select training mode
    parser.add_argument('--training_mode', type=str, default='None',
                        help="Select from sl, ssl, both")

    # Add argument to select results
    parser.add_argument('--plots', type=bool, default=True)
    parser.add_argument('--tables', type=bool, default=False)

    args = parser.parse_args()

    if args.final_only and args.training_mode == 'None':
        # Results of SL model
        print("*" * 10, "SL Results", "*" * 10)
        print("Using Random Forest with ADASYN Over Sampling")
        model = load_model('RandomForestClassifier_ADASYN.pkl',
                           training_mode='sl')
        train_metrics = get_model_metrics(model, X_train, y_train)
        test_metrics = get_model_metrics(model, X_test, y_test)
        print(f"Training metrics: {train_metrics}")
        print(f"Testing metrics: {test_metrics}")

        # Results of SSL model
        print("*"*10, "SSL Results", "*"*10)
        print("Using Self Training with Random Forest as Base Classifier")
        model = load_model('SelfTraining_RandomForestClassifier.pkl',
                           training_mode='ssl')
        train_metrics = get_model_metrics(model, X_train, y_train)
        test_metrics = get_model_metrics(model, X_test, y_test)
        print(f"Training metrics: {train_metrics}")
        print(f"Testing metrics: {test_metrics}")
        sys.exit()

    if args.training_mode == 'sl':
        if args.plots:
            generate_sl_results()
        if args.tables:
            generate_sl_tables()
    elif args.training_mode == 'ssl':
        if args.plots:
            generate_ssl_results()
        if args.tables:
            generate_ssl_tables()
    elif args.training_mode == 'both':
        if args.plots:
            generate_sl_results()
            generate_ssl_results()
        if args.tables:
            generate_sl_tables()
            generate_ssl_tables()
    else:
        raise Exception("Select training mode from sl, ssl, both")
    pass
