import time
import csv
import os
import numpy as np
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn

from warm_start.warm_start import WarmStart
from warm_start.experience import Experience, ExperienceStore
from warm_start.my_text_classification_extractor import MyTextClassificationFeatureExtractor

seeds = [0]
search_iterations = [5]
output_csv = "experiment_20news_warm.csv"
first_time = not os.path.exists(output_csv)

if first_time:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain","n_samples","n_features","n_classes","categories",
            "random_state","search_iters","best_score","time_seconds","best_pipeline","warm_start"
        ])

categories = ["comp.graphics", "sci.med"]
newsgroups_data = fetch_20newsgroups(subset='train', categories=categories)
X_text = newsgroups_data.data
y = np.array(newsgroups_data.target)

vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_matrix = vectorizer.fit_transform(X_text).toarray()

n_samples = len(X_text)
n_features = X_matrix.shape[1]
n_classes = len(set(y))
domain = "20newsgroups_warm_subset"

print(f"[WARM] docs={n_samples}, shape={X_matrix.shape}, classes={n_classes}")

warm_start = WarmStart(
    positive_min_threshold=0.2,
    k_pos=20,
    k_neg=20,
    max_alpha=0.05,
    min_alpha=-0.02,
    from_date=None,
    to_date=None,
    include=None,
    exclude=None,
)

warm_start.pre_warm_up(X_text, y)

def my_generator_fn(sampler):
    # Solo NaiveBayes para garantizar ligereza
    classifier = sampler.choice(["NaiveBayes"])

for seed in seeds:
    for iters in search_iterations:
        start_time = time.time()

        exclude_list = [
            "CountVectorizer","TfidfVectorizer","HashingVectorizer",
            "KernelPCA","KernelCenterer","AdditiveChi2Sampler","Nystroem",
            "FastICA","KBinsDiscretizer","SelectKBest","SelectFdr","SelectFwe",
            "SelectPercentile","MiniBatchKMeans","FeatureAgglomeration",
            "SpectralClustering","TruncatedSVD","NMF","GaussianProcessClassifier",
            "GradientBoostingClassifier","ExtraTreesClassifier","MLPClassifier",
            "SVC","RandomForestClassifier","MultinomialNB"
        ]
        exclude_pattern = "(" + "|".join(exclude_list) + ")"

        registry = find_classes([sklearn], exclude=exclude_pattern)

        warm_start_model = warm_start.warm_up(my_generator_fn)

        automl = AutoML(
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=registry,
            random_state=seed,
            search_iterations=iters,
            search_timeout=60,
            initial_model=warm_start_model
        )

        automl.fit(X_matrix, y)
        elapsed = time.time() - start_time

        best_score = automl.best_scores_[0][0]
        best_pipeline = automl.best_pipelines_[0]

        print(f"[WARM] seed={seed}, iters={iters} => {best_score:.4f}, time={elapsed:.2f}s")

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                domain,n_samples,n_features,n_classes,str(newsgroups_data.target_names),
                seed,iters,best_score,f"{elapsed:.2f}",str(best_pipeline),1
            ])

print("\n[WARM] ¡Terminado! Revisa experiment_20news_warm.csv")
