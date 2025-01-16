import argparse
import time
import csv
import os
import numpy as np

from autogoal.search import RichLogger
from autogoal.ml import AutoML
from autogoal.meta_learning import WarmStart
from autogoal.meta_learning._logging import ExperienceLogger
from autogoal.datasets import (
    imdb_50k_movie_reviews,
    ag_news,
    yelp_reviews,
    # 20 newsgroups no viene en `autogoal.datasets`, ***cargarlo manualmente****
)
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from sklearn.feature_extraction.text import TfidfVectorizer
from autogoal.search._warm_start_pge import NSPEWarmStartSearch
from autogoal.meta_learning.feature_extraction.text_classification import TextClassificationFeatureExtractor

#base config
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imdb",
                    help="Which dataset to run warm start on: imdb, agnews, yelp, or 20ng")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iters", type=int, default=10)
args = parser.parse_args()

DATASET = args.dataset.lower()
SEED = args.seed
SEARCH_ITERS = args.iters

output_csv = f"experiment_{DATASET}_warmstart.csv"
first_time = not os.path.exists(output_csv)
if first_time:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "warm_alias", 
            "n_samples",
            "n_features",
            "n_classes",
            "seed",
            "search_iters",
            "best_score",
            "time_seconds",
            "best_pipeline"
        ])

# Cargado y vectorizado
def load_and_vectorize(dataset_name):
    if dataset_name == "imdb":
        print("[WARMSTART] Cargando IMDB dataset...")
        X_train, y_train, X_test, y_test = imdb_50k_movie_reviews.load()
        X_text = X_train + X_test
        y = np.concatenate([y_train, y_test])
        alias = "imdb"

    elif dataset_name == "agnews":
        print("[WARMSTART] Cargando AGNews dataset...")
        X_train, y_train, X_test, y_test = ag_news.load(True) 
        X_text = X_train + X_test
        y = np.concatenate([y_train, y_test])
        alias = "agnews"

    elif dataset_name == "yelp":
        print("[WARMSTART] Cargando Yelp dataset...")
        X_train, y_train, X_test, y_test = yelp_reviews.load(True)
        X_text = X_train + X_test
        y = np.concatenate([y_train, y_test])
        alias = "yelp"

    elif dataset_name == "20ng":
        print("[WARMSTART] Cargando 20 Newsgroups dataset...")
        from sklearn.datasets import fetch_20newsgroups
        data_20ng = fetch_20newsgroups(subset='train')
        X_text = data_20ng.data
        y = np.array(data_20ng.target)
        alias = "20newsgroups"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_matrix = vectorizer.fit_transform(X_text).toarray()

    return X_text, y, X_matrix, alias

X_text, y, X_matrix, dataset_alias = load_and_vectorize(DATASET)

n_samples = len(X_text)
n_classes = len(set(y))
n_features = X_matrix.shape[1]
print(f"[WARMSTART] Dataset={dataset_alias}, n_samples={n_samples}, n_features={n_features}, n_classes={n_classes}")

#meatafeatures extractor
extractor = TextClassificationFeatureExtractor()
dataset_feat = extractor.extract_features(X_text, y)

#warm start config
warm = WarmStart(
    positive_min_threshold=0.2,
    k_pos=20,
    k_neg=20,
    max_alpha=0.05,
    min_alpha=-0.02,
    dataset_feature_extractor=TextClassificationFeatureExtractor,
    system_feature_extractor=None,  # cambiar e importar las funciones necesarias en caso de que se use
    include=".*",          # incluir todos 
    exclude=f"({dataset_alias})",   # Excluir el alias actual
)

# pre_warm_up => asigna X_train, y_train:-------  extrae metafeatures
warm.pre_warm_up(X_matrix, y)

# info del espacio de muestreo
def generator_fn(sampler):
    pass

#filtra y carga las experiencias + model ajuste
print(f"[WARMSTART] Cargando experiences, EXCLUYENDO alias={dataset_alias} ...")
warm.warm_up(generator_fn)

exclude_pattern = (
    "("
    "CountVectorizer|TfidfVectorizer|HashingVectorizer|"
    "KernelPCA|KernelCenterer|AdditiveChi2Sampler|Nystroem"
    ")"
)
my_registry = find_classes(modules=[sklearn], exclude=exclude_pattern)

# acá se crea un ExperienceLogger, pasando el dataset_features
warm_alias = f"{dataset_alias}_warm"
from autogoal.meta_learning._logging import ExperienceLogger
experience_logger = ExperienceLogger(
    alias=warm_alias,
    dataset_feature_extractor_name="TextClassificationFeatureExtractor",
    system_feature_extractor_name="NoSystem",
)

start_time = time.time()

automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    registry=my_registry,
    random_state=SEED,
    search_iterations=SEARCH_ITERS,
)

# se usan los search
from autogoal.search._warm_start_pge import NSPEWarmStartSearch
automl.search_algorithm = NSPEWarmStartSearch
automl.search_kwargs["warm_start"] = warm  # le pasamos la clase WarmStart

automl.fit(X_matrix, y, logger=[experience_logger])
elapsed = time.time() - start_time

best_score = automl.best_scores_[0]
best_pipeline = automl.best_pipelines_[0]

print(f"[WARMSTART] Dataset={dataset_alias}, seed={SEED}, iters={SEARCH_ITERS} => Score={best_score[0]:.4f}, time={elapsed:.2f}s")

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        warm_alias,
        n_samples,
        n_features,
        n_classes,
        SEED,
        SEARCH_ITERS,
        best_score[0],
        f"{elapsed:.2f}",
        str(best_pipeline)
    ])

print("\n[WARMSTART] Finalizado.")
print(f"Revisa CSV: {output_csv}")
