import time
import csv
import os
import numpy as np

from sklearn.datasets import fetch_rcv1
from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn


subset = 'train'  # RCV1-subset => "train" (~23k docs)
seeds = [0, 1]
search_iterations = [10, 50]

output_csv = f"experiment_rcv1_phase1_{subset}.csv"
first_time = not os.path.exists(output_csv)
if first_time:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain",
            "subset",
            "n_samples",
            "n_features",
            "n_classes",
            "random_state",
            "search_iters",
            "best_score",
            "time_seconds",
            "best_pipeline"
        ])

domain_name = "rcv1_offline"

print(f"Cargando RCV1(subset='{subset}') localmente (sin descarga)...")
rcv1 = fetch_rcv1(subset=subset, download_if_missing=False)
X_sparse = rcv1.data
Y_multilabel = rcv1.target

n_samples = X_sparse.shape[0]
n_features = X_sparse.shape[1]
n_labels = Y_multilabel.shape[1]

print(f"RCV1 {subset}: {X_sparse.shape}, {n_labels} labels (multilabel)")

y_single = np.full(n_samples, -1, dtype=int)
for i in range(n_samples):
    active_labels = Y_multilabel[i].nonzero()[1]  
    if len(active_labels) > 0:
        y_single[i] = active_labels[0]

mask = y_single != -1
X_sparse = X_sparse[mask]
y_single = y_single[mask]
X_array = X_sparse.toarray()
n_samples_final = X_array.shape[0]
n_features_final = X_array.shape[1]
n_classes_final = len(set(y_single))

print(f"{n_samples_final} docs, {n_features_final} features, {n_classes_final} single-label classes")

exclude_pattern = (
    "("
    "CountVectorizer|TfidfVectorizer|HashingVectorizer|"
    "KernelPCA|KernelCenterer|AdditiveChi2Sampler|Nystroem"
    ")"
)

for seed in seeds:
    for iters in search_iterations:
        start_time = time.time()

        my_registry = find_classes(modules=[sklearn], exclude=exclude_pattern)

        automl = AutoML(
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=my_registry,
            random_state=seed,
            search_iterations=iters
        )

        automl.fit(X_array, y_single)
        elapsed = time.time() - start_time

        best_score = automl.best_scores_[0]   # (score,)
        best_pipeline = automl.best_pipelines_[0]

        print(f"[RCV1-{subset}] seed={seed}, iters={iters} => Score={best_score[0]:.4f}, Time={elapsed:.2f}s")

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                domain_name,
                subset,
                n_samples_final,
                n_features_final,
                n_classes_final,
                seed,
                iters,
                f"{best_score[0]:.4f}",
                f"{elapsed:.2f}",
                str(best_pipeline)
            ])

print(f"\nExperimento RCV1({subset}) finalizado. Revisa: {output_csv}")
