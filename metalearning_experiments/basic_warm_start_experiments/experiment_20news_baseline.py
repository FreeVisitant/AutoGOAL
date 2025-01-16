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

from warm_start.experience import Experience, ExperienceStore
from warm_start.my_text_classification_extractor import MyTextClassificationFeatureExtractor

categories = ["comp.graphics", "sci.med"]

seeds = [0]
search_iterations = [5]
output_csv = "experiment_20news_baseline.csv"
first_time = not os.path.exists(output_csv)

EXPERIENCES_DIR = "experiences_logs"
os.makedirs(EXPERIENCES_DIR, exist_ok=True)

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

newsgroups_data = fetch_20newsgroups(subset='train', categories=categories)
X_text = newsgroups_data.data
y = np.array(newsgroups_data.target)

vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_matrix = vectorizer.fit_transform(X_text).toarray()

n_samples = len(X_text)
n_features = X_matrix.shape[1]
n_classes = len(set(y))
domain = "20newsgroups_baseline_subset"

print(f"[BASELINE] docs={n_samples}, shape={X_matrix.shape}, classes={n_classes}")

exclude_list = [
    "CountVectorizer",
    "TfidfVectorizer",
    "HashingVectorizer",
    "KernelPCA",
    "KernelCenterer",
    "AdditiveChi2Sampler",
    "Nystroem",
    "FastICA",
    "KBinsDiscretizer",
    "SelectKBest",
    "SelectFdr",
    "SelectFwe",
    "SelectPercentile",
    "MiniBatchKMeans",
    "FeatureAgglomeration",
    "SpectralClustering",
    "TruncatedSVD",
    "NMF",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    "ExtraTreesClassifier",
    "MLPClassifier",
    "SVC",
    "RandomForestClassifier",
    "MultinomialNB"
]
exclude_pattern = "(" + "|".join(exclude_list) + ")"

for seed in seeds:
    for iters in search_iterations:
        start_time = time.time()

        registry = find_classes(modules=[sklearn], exclude=exclude_pattern)

        automl = AutoML(
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=registry,
            random_state=seed,
            search_iterations=iters,
            search_timeout=60  
        )

        automl.fit(X_matrix, y)
        elapsed = time.time() - start_time

        best_score = automl.best_scores_[0][0]
        best_pipeline = automl.best_pipelines_[0]

        print(f"[BASELINE] seed={seed}, iters={iters} => score={best_score:.4f}, time={elapsed:.2f}s")

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
                best_score,
                f"{elapsed:.2f}",
                str(best_pipeline)
            ])

        extractor = MyTextClassificationFeatureExtractor()
        dataset_features = extractor.extract_features(X_text, y)
        system_features = np.array([0.0])

        exp = Experience(
            dataset_feature_extractor_name=extractor.__class__.__name__,
            system_feature_extractor_name="DummySystemExtractor",
            dataset_features=dataset_features,
            system_features=system_features,
            f1=best_score,
            evaluation_time=elapsed,
            alias=f"20news_baseline_seed{seed}_iters{iters}",
            algorithms=[{"pipeline": str(best_pipeline)}]
        )

        pkl_filename = f"20news_baseline_seed{seed}_iters{iters}.pkl"
        pkl_path = os.path.join(EXPERIENCES_DIR, pkl_filename)
        with open(pkl_path, "wb") as pf:
            pickle.dump(exp, pf)
        print(f"Guardada experiencia en: {pkl_path}")

print("\n[BASELINE] ¡Terminado! Revisa experiment_20news_baseline.csv y experiences_logs/*.pkl")
