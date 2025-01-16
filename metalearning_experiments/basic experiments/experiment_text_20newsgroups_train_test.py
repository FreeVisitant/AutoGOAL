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

categories = None  
seeds = [0, 1]     
search_iterations = [10, 50]  

output_csv = "experiment_text_20newsgroups_train_test.csv"
first_time = not os.path.exists(output_csv)

if first_time:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain",
            "n_train_samples",
            "n_test_samples",
            "n_features",
            "n_classes",
            "random_state",
            "search_iters",
            "train_score",
            "test_score",
            "time_seconds",
            "best_pipeline"
        ])

train_data = fetch_20newsgroups(subset='train', categories=categories)
X_train_text = train_data.data
y_train = np.array(train_data.target)

test_data = fetch_20newsgroups(subset='test', categories=categories)
X_test_text = test_data.data
y_test = np.array(test_data.target)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_matrix = vectorizer.fit_transform(X_train_text).toarray()
X_test_matrix  = vectorizer.transform(X_test_text).toarray()

n_train_samples = len(X_train_text)
n_test_samples  = len(X_test_text)
n_features = X_train_matrix.shape[1]
n_classes = len(set(y_train))
domain = "text_20newsgroups_train_test"

print(f"Train docs: {n_train_samples}, Test docs: {n_test_samples}, "
      f"TF-IDF features: {n_features}, #classes={n_classes}")

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

        automl.fit(X_train_matrix, y_train)
        elapsed = time.time() - start_time

        train_score = automl.best_scores_[0] 
        best_pipeline = automl.best_pipelines_[0]
        best_pipeline.send("eval")

        dummy_y = np.zeros_like(y_test) 
        y_pred = best_pipeline.run(X_test_matrix, dummy_y)  

        test_score = np.mean(y_pred == y_test)

        print(f"[20NG] seed={seed}, iters={iters} -> "
              f"Train={train_score[0]:.4f}, Test={test_score:.4f}, Time={elapsed:.2f}s")

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                domain,
                n_train_samples,
                n_test_samples,
                n_features,
                n_classes,
                seed,
                iters,
                train_score[0],
                f"{test_score:.4f}",
                f"{elapsed:.2f}",
                str(best_pipeline)
            ])

print(f"\nExperimento 20NG train/test finalizado. Revisa: {output_csv}")
