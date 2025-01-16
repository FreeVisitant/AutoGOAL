import time
import csv
import os
import numpy as np

from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.meta_learning._logging import ExperienceLogger
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn

from autogoal.datasets import ag_news
from sklearn.feature_extraction.text import TfidfVectorizer

# config, aumentar iteraciones y seeds para obtener más salidas
seeds = [0, 1]
search_iterations_list = [10, 50] 

output_csv = "experiment_agnews_baseline.csv"
first_time = not os.path.exists(output_csv)

if first_time:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain",
            "n_samples",
            "n_features",
            "n_classes",
            "random_state",
            "search_iters",
            "best_score",
            "time_seconds",
            "best_pipeline"
        ])

# Cargado y vectorizado
print("Cargando AG News...")
X_train, y_train, X_test, y_test = ag_news.load(True)

# combinar para mayor set
X_text = X_train + X_test
y = np.concatenate([y_train, y_test])

n_samples = len(X_text)
n_classes = len(set(y))

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=3000
)
X_matrix = vectorizer.fit_transform(X_text).toarray()
n_features = X_matrix.shape[1]

print(f"Documentos: {n_samples} | TF-IDF shape: {X_matrix.shape} | n_classes={n_classes}")

#########################
#Extracción de metafeatures
#########################
# Función genera un vector con metafeatures:
# - n_instances, n_classes
# - longitud media y std de los docs
# - entropía de clases, etc.
def compute_simple_metafeatures(X_text, y):
    # info de dataset
    n_inst = len(X_text)
    unique_classes = np.unique(y) if y is not None else []
    n_cl = len(unique_classes)
    # Distribución de clases
    class_counts = np.array([np.sum(y == c) for c in unique_classes]) if n_cl>0 else []
    class_probs = class_counts / n_inst if n_inst>0 else []
    # entropía
    class_entropy = -np.sum(class_probs * np.log2(class_probs + 1e-10)) if n_cl>1 else 0

    # min y max prob
    if n_cl>0:
        min_class_prob = np.min(class_probs)
        max_class_prob = np.max(class_probs)
    else:
        min_class_prob, max_class_prob = 0,0

    imbalance_ratio = 0
    if n_cl>1:
        imbalance_ratio = min_class_prob / (max_class_prob + 1e-10)

    # 2) Longitud del doc
    doc_lengths = np.array([len(doc) for doc in X_text]) if n_inst>0 else np.array([0])
    avg_len = np.mean(doc_lengths)
    std_len = np.std(doc_lengths)
    coef_var_len = std_len / (avg_len + 1e-10) if avg_len>0 else 0

    # agrupamos los elmentos en un vector
    features = np.array([
        n_inst,
        n_cl,
        class_entropy,
        min_class_prob,
        max_class_prob,
        imbalance_ratio,
        avg_len,
        std_len,
        coef_var_len
    ], dtype=np.float32)

    return features

# se generan los metafeatures
dataset_features = compute_simple_metafeatures(X_text, y)

#los excludes que dan problema
exclude_pattern = (
    "("
    "CountVectorizer|TfidfVectorizer|HashingVectorizer|"
    "KernelPCA|KernelCenterer|AdditiveChi2Sampler|Nystroem"
    ")"
)
my_registry = find_classes(
    modules=[sklearn],
    exclude=exclude_pattern
)
#experimentanción en base a seed*iterations
domain = "agnews"
for seed in seeds:
    for search_iters in search_iterations_list:
        start_time = time.time()

        # acá se crea un ExperienceLogger, pasando el dataset_features
        experience_logger = ExperienceLogger(
            dataset_features=dataset_features,
            system_features=None,
            dataset_feature_extractor_name="SimpleMetaFeats",
            system_feature_extractor_name="NoSystem",
            alias=domain
        )

        automl = AutoML(
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=my_registry,
            random_state=seed,
            search_iterations=search_iters
        )

        # ajuste y registrro de logs
        automl.fit(X_matrix, y, logger=[experience_logger])
        elapsed = time.time() - start_time

        best_score = automl.best_scores_[0]
        best_pipeline = automl.best_pipelines_[0]

        print(f"[AGNEWS] SEED={seed} | ITERS={search_iters} -> Score={best_score[0]:.4f} | Time={elapsed:.2f}s")

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                domain,
                n_samples,
                n_features,
                n_classes,
                seed,
                search_iters,
                best_score[0],
                f"{elapsed:.2f}",
                str(best_pipeline)
            ])

print("\nExperimento BASELINE AGNews finalizado.")
print(f"Revisa: {output_csv}")
print(f"Experiencias guardadas en ~/.autogoal/experience_store/{domain}/... (alias='{domain}')")
