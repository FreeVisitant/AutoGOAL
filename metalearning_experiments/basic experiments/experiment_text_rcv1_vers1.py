import time
import csv
import os
import numpy as np

from sklearn.datasets import fetch_rcv1
from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn

max_docs = 10000  
seeds = [0, 1]
search_iterations = [10, 50]
output_csv = "experiment_text_rcv1_vers1.csv"

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

print("Cargando RCV1 (subset='train')... .")
rcv1 = fetch_rcv1(subset='train')  

X_sparse = rcv1.data
Y_multilabel = rcv1.target
n_samples_total = X_sparse.shape[0]
n_features_total = X_sparse.shape[1]
n_labels = Y_multilabel.shape[1]
print(f"RCV1 shape: {X_sparse.shape}, #labels={n_labels}")

X_sparse = X_sparse[:max_docs]
Y_multilabel = Y_multilabel[:max_docs]

y_single = np.full(X_sparse.shape[0], -1, dtype=int)
for i in range(X_sparse.shape[0]):
    row_labels = Y_multilabel[i].nonzero()[1] 
    if len(row_labels) > 0:
        y_single[i] = row_labels[0]
mask_valid = y_single != -1
X_sparse = X_sparse[mask_valid]
y_single = y_single[mask_valid]

X_array = X_sparse.toarray()  
n_samples = X_array.shape[0]
n_features = X_array.shape[1]
n_classes = len(set(y_single))
print(f"Tras filtrar docs sin etiqueta: {n_samples} muestras, {n_features} features, {n_classes} clases (pseudo single-label).")

domain = "text_rcv1"

exclude_pattern = (
    "("
    "CountVectorizer|TfidfVectorizer|HashingVectorizer|"  # excluye text transforms
    "KernelPCA|KernelCenterer|AdditiveChi2Sampler|Nystroem"  # excluye kernel-based NxN
    ")"
)

for seed in seeds:
    for iters in search_iterations:
        start = time.time()
        my_registry = find_classes(modules=[sklearn], exclude=exclude_pattern)
        
        automl = AutoML(
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=my_registry,
            random_state=seed,
            search_iterations=iters
        )
        
        automl.fit(X_array, y_single)
        elapsed = time.time() - start
        
        best_score = automl.best_scores_[0]  
        best_pipeline = automl.best_pipelines_[0]

        print(f"RCV1, seed={seed}, iters={iters} => Score={best_score[0]:.4f}, time={elapsed:.2f}s")

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                domain,
                n_samples,
                n_features,
                n_classes,
                seed,
                iters,
                best_score[0],
                f"{elapsed:.2f}",
                str(best_pipeline)
            ])

print("\nExperimento RCV1 finalizado. Revisa:", output_csv)
