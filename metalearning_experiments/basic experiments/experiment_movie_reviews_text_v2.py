import os
import csv
import time
import numpy as np

import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer


from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn

seeds = [0, 1, 2, 3]
search_iterations = [10, 50, 100, 200]
output_csv = "/home/coder/autogoal/results/experiment_movie_reviews_text_v2.csv"
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

domain_name = "nltk_movie_reviews_heavy"

file_ids = movie_reviews.fileids()  # ~2000 docs
texts = []
labels = []

for fid in file_ids:
    category = movie_reviews.categories(fid)[0]  # 'pos' o 'neg'
    text = movie_reviews.raw(fid)
    texts.append(text)
    labels.append(category)

label_map = {'neg': 0, 'pos': 1}
y = np.array([label_map[lbl] for lbl in labels], dtype=int)

n_samples = len(texts)
n_classes = len(set(labels))

print(f"Movie Reviews: {n_samples} docs, {n_classes} classes (pos/neg)")

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_matrix = vectorizer.fit_transform(texts).toarray()
n_features = X_matrix.shape[1]
print(f"TF-IDF shape: {X_matrix.shape} with max_features=10000")

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

        automl.fit(X_matrix, y)
        elapsed = time.time() - start_time

        best_score = automl.best_scores_[0]
        best_pipeline = automl.best_pipelines_[0]

        print(f"[MovieReviews-Heavy] seed={seed}, iters={iters} => "
              f"Score={best_score[0]:.4f}, Time={elapsed:.2f}s")

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

print(f"\nExperimento 'heavier' en Movie Reviews finalizado. Revisa: {output_csv}")
