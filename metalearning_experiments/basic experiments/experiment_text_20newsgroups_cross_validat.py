import os
import csv
import time
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import KFold

from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn

categories = [
    'comp.graphics', 
    'sci.med',
    'rec.sport.baseball',
    'talk.politics.guns',
    'soc.religion.christian',
    'misc.forsale'
]

k_folds = 3              # número de folds
seeds = [0]              
search_iterations = [10] 

output_csv = "experiment_text_cv.csv"
first_time = not os.path.exists(output_csv)

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)
    if first_time:
        writer.writerow([
            "domain",       # "text"
            "n_samples",
            "n_features",
            "n_classes",
            "categories",
            "n_folds",
            "random_state",
            "search_iters",
            "fold_index",
            "fold_score",
            "fold_time_s",
            "best_pipeline",
            "avg_score_across_folds"
        ])
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
X_text = newsgroups_train.data  
y = np.array(newsgroups_train.target)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_matrix = vectorizer.fit_transform(X_text).toarray()

n_samples = len(X_text)
n_features = X_matrix.shape[1]
n_classes = len(set(y))
domain = "text"

print(f"Documentos: {n_samples} | TF-IDF shape: {X_matrix.shape} | n_classes={n_classes}")

for seed in seeds:
    for iters in search_iterations:
        fold_scores = []

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        fold_index = 0
        for train_index, test_index in kf.split(X_matrix):
            fold_index += 1

            X_train_fold = X_matrix[train_index]
            y_train_fold = y[train_index]
            X_test_fold  = X_matrix[test_index]
            y_test_fold  = y[test_index]

            start_time = time.time()
            registry = find_classes(modules=[sklearn])
            automl = AutoML(
                input=(MatrixContinuousDense, Supervised[VectorCategorical]),
                output=VectorCategorical,
                registry=registry,
                random_state=seed,
                search_iterations=iters
            )

            automl.fit(X_train_fold, y_train_fold)
            fold_time = time.time() - start_time
            best_pipeline = automl.best_pipelines_[0]
            best_pipeline.send("eval")
            y_pred = best_pipeline.run((X_test_fold, None))
            fold_score = np.mean(y_pred == y_test_fold)
            fold_scores.append(fold_score)

            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    domain,
                    n_samples,
                    n_features,
                    n_classes,
                    str(categories),
                    k_folds,
                    seed,
                    iters,
                    fold_index,
                    f"{fold_score:.4f}",
                    f"{fold_time:.2f}",
                    str(best_pipeline),
                    0  
                ])

            print(f"[Fold {fold_index}/{k_folds}] seed={seed}, iters={iters} -> FoldScore={fold_score:.4f}, Time={fold_time:.2f}s")

        # Calculamos la media de los folds
        avg_score = np.mean(fold_scores)

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                domain,
                n_samples,
                n_features,
                n_classes,
                str(categories),
                k_folds,
                seed,
                iters,
                "AVG",  # indica promedio
                f"{avg_score:.4f}",
                "--",   # no time
                "None", # no pipeline
                f"{avg_score:.4f}"
            ])

        print(f"Promedio final en {k_folds} folds (seed={seed}, iters={iters}): {avg_score:.4f}")

print(f"\nExperimento cross-validation finalizado. Revisa: {output_csv}")
