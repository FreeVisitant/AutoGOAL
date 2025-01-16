import time
import csv
import os
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn

# config, aumentar iteraciones y seeds para obtener más salidas
categories = None  # todas las categorías
seeds = [0, 1]     
search_iterations = [10, 50]

output_csv = "experiment_text_20newsgroups_v3"
first_time = not os.path.exists(output_csv)

if first_time:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain",          
            "n_samples",
            "n_features",
            "n_classes",
            "categories",
            "random_state",
            "search_iters",
            "best_score",
            "time_seconds",
            "best_pipeline"
        ])

# Cargado y vectorizado
newsgroups_data = fetch_20newsgroups(subset='train', categories=categories)
X_text = newsgroups_data.data
y = np.array(newsgroups_data.target)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_matrix = vectorizer.fit_transform(X_text).toarray()

n_samples = len(X_text)
n_features = X_matrix.shape[1]
n_classes = len(set(y))
domain = "text"

print(f"Documentos: {n_samples} | TF-IDF shape: {X_matrix.shape} | n_classes={n_classes}")
print("Categories:", newsgroups_data.target_names)

for seed in seeds:
    for iters in search_iterations:
        start_time = time.time()

        exclude_pattern = (
            "("
            "CountVectorizer|TfidfVectorizer|HashingVectorizer|"  # text transf
            "KernelPCA|KernelCenterer|AdditiveChi2Sampler|Nystroem"  
            ")"
        )

        my_registry = find_classes(
            modules=[sklearn],
            exclude=exclude_pattern
        )

        automl = AutoML(
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=my_registry,
            random_state=seed,
            search_iterations=iters
        )

        # Training
        automl.fit(X_matrix, y)
        elapsed = time.time() - start_time

        # Mejor pipeline
        best_score = automl.best_scores_[0]
        best_pipeline = automl.best_pipelines_[0]

        print(f"Seed={seed}, iters={iters} -> Score={best_score[0]:.4f}, Time={elapsed:.2f}s")

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                domain,
                n_samples,
                n_features,
                n_classes,
                str(newsgroups_data.target_names),
                seed,
                iters,
                best_score[0],
                f"{elapsed:.2f}",
                str(best_pipeline)
            ])

print("\nExperimentos finalizados. Revisa:", output_csv)
