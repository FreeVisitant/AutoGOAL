import os
import csv
import time
import numpy as np
import random

import nltk
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer

from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn

seeds = [0, 1]
search_iterations = [10, 50]
output_csv = "/home/coder/autogoal/results/experiment_reuters_v1_text_nltk.csv"
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

domain_name = "nltk_reuters_singlelabel"

def load_reuters_single_label():
    docs = []
    labels = []

    fileids = reuters.fileids()
    for fid in fileids:
        cats = reuters.categories(fid)  # lista de etiquetas
        if len(cats) == 0:
            continue
        # Tomamos la primera etiqueta como "clase"
        label = cats[0]
        text = reuters.raw(fid)
        docs.append(text)
        labels.append(label)

    # Mezclamos
    data = list(zip(docs, labels))
    random.shuffle(data)
    X, y = zip(*data)
    return list(X), list(y)

def run_experiment():
    # 1) Cargamos docs y etiquetas single-label
    X_text, y_labels = load_reuters_single_label()
    
    # Convertimos las etiquetas a enteros
    unique_labels = sorted(set(y_labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_int = [label_to_idx[lbl] for lbl in y_labels]

    n_samples = len(X_text)
    n_classes = len(unique_labels)

    print(f"Reuters single-label: {n_samples} docs, {n_classes} classes.")

    # 2) Vectorizamos (TF-IDF) con un max_features moderado
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X_matrix = vectorizer.fit_transform(X_text).toarray()
    n_features = X_matrix.shape[1]
    print(f"TF-IDF shape: {X_matrix.shape} => {n_samples} x {n_features}")

    exclude_pattern = "(" + "|".join([
        "CountVectorizer","TfidfVectorizer","HashingVectorizer",
        "KernelPCA","KernelCenterer","AdditiveChi2Sampler","Nystroem"
    ]) + ")"

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

            automl.fit(X_matrix, y_int)
            elapsed = time.time() - start_time

            best_score = automl.best_scores_[0]
            best_pipeline = automl.best_pipelines_[0]

            print(f"[Reuters] seed={seed}, iters={iters} => Score={best_score[0]:.4f}, Time={elapsed:.2f}s")

            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    domain_name,
                    n_samples,
                    n_features,
                    n_classes,
                    seed,
                    iters,
                    f"{best_score[0]:.4f}",
                    f"{elapsed:.2f}",
                    str(best_pipeline)
                ])

    print(f"\nExperimento con Reuters finalizado. Revisa: {output_csv}")

if __name__ == "__main__":
    run_experiment()
