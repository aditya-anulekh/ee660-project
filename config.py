import os

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
RANDOM_STATE = 42
N_FOLDS = 5

# Project specific config
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
