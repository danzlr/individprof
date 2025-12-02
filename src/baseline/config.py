"""
Configuration file for the NTO ML competition baseline.
"""

from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants

#DIRECTORIES
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# REQUIRED OLD VARS (чтобы train.py и predict.py работали)


EARLY_STOPPING_ROUNDS = 100
MODEL_FILENAME = "lgb_model.txt"   # predict.py ищет это

# GENERAL SETTINGS
RANDOM_STATE = 42
TARGET = constants.COL_TARGET

# Temporal split ratio (80% train / 20% val)
TEMPORAL_SPLIT_RATIO = 0.8



# TF-IDF SETTINGS

TFIDF_MAX_FEATURES = 10000      # было 500 качество сильно выше
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# BERT SETTINGS
BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 8
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768
BERT_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
BERT_GPU_MEMORY_FRACTION = 0.75

# CATEGORICAL FEATURES

CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]



# LIGHTGBM SETTINGS
LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "boosting_type": "gbdt",

    "num_leaves": 63,            # лучше, чем 31
    "learning_rate": 0.008,
    "n_estimators": 3000,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,

    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "min_child_samples": 20,

    "seed": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

from lightgbm import early_stopping

LGB_FIT_PARAMS = {
    "eval_metric": "rmse",
    "callbacks": [early_stopping(EARLY_STOPPING_ROUNDS)],
}
