import time
import csv
import os
import numpy as np
import random
import sys
import logging
import re


from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.meta_learning._logging import ExperienceLogger
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn


from sklearn.feature_extraction.text import TfidfVectorizer


from autogoal.meta_learning.feature_extraction.text_classification import TextClassificationFeatureExtractor


import nltk
try:
    nltk.data.find('corpora/inaugural')
    logging.info("El corpus 'inaugural' ya está descargado.")
except LookupError:
    logging.info("Descargando el corpus 'inaugural' de NLTK...")
    nltk.download('inaugural')

from nltk.corpus import inaugural


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ExperienceLoggerWithEvalTime(ExperienceLogger):
    def eval_solution(self, solution, fitness, observations):

        if hasattr(fitness, '__len__') and len(fitness) > 1:
            f1_score = fitness[0]
            evaluation_time = fitness[1]
        else:
            f1_score = fitness
            evaluation_time = 0.0  
        
        while isinstance(f1_score, (list, tuple)) and len(f1_score) == 1:
            f1_score = f1_score[0]


        print(f"[DEBUG] Registrando experiencia: f1_score = {f1_score}, evaluation_time = {evaluation_time}")

        accuracy = None
        if observations is not None and "Accuracy" in observations:
            accuracy = observations["Accuracy"]
        
        self.log_experience(
            solution=solution,
            f1_score=f1_score,
            evaluation_time=evaluation_time,
            accuracy=accuracy,
            error=None,
        )


seeds = [0, 1]
search_iterations_list = [10, 50]

output_csv = "experiment_inaugural_baseline.csv"
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

print("Cargando corpus Inaugural y generando etiquetas...")


def label_from_fileid(fid):
    match = re.match(r"(\d{4})-", fid)
    if match:
        year = int(match.group(1))
        return "early" if year < 1900 else "late"
    else:
        return "unknown"

fileids = inaugural.fileids()
documents = []
for fid in fileids:
    text = inaugural.raw(fid)
    label = label_from_fileid(fid)
    if label in {"early", "late"}:
        documents.append((text, label))


random.shuffle(documents)
X_all, y_all = zip(*documents)
X_all = list(X_all)
y_all = list(y_all)


X_text = X_all
y = y_all

n_samples = len(X_text)
n_classes = len(set(y))

print(f"Documentos: {n_samples} | Número de clases: {n_classes}")
print("Clases:", list(set(y)))


vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=3000,
    ngram_range=(1, 2)
)
X_matrix = vectorizer.fit_transform(X_text).toarray()
n_features = X_matrix.shape[1]

print(f"TF-IDF shape: {X_matrix.shape}")


text_features_extractor = TextClassificationFeatureExtractor()
dataset_features = text_features_extractor.extract_features(X_text, y)

domain = "inaugural"


for seed in seeds:
    for search_iters in search_iterations_list:
        start_time = time.time()

        experience_logger = ExperienceLoggerWithEvalTime(
            dataset_features=dataset_features,
            system_features=np.array([]),
            dataset_feature_extractor_name="TextClassificationFeatureExtractor",
            system_feature_extractor_name="NoSystem",
            alias=domain
        )

        my_registry = find_classes(
            modules=[sklearn],
            exclude="(CountVectorizer|TfidfVectorizer|HashingVectorizer|KernelPCA|KernelCenterer|AdditiveChi2Sampler|Nystroem)"
        )

        automl = AutoML(
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=my_registry,
            random_state=seed,
            search_iterations=search_iters
        )
        automl.fit(X_matrix, y, logger=[experience_logger])
        elapsed = time.time() - start_time

        best_score = automl.best_scores_[0]  
        best_pipeline = automl.best_pipelines_[0]

        print(f"[{domain}] SEED={seed} | ITERS={search_iters} -> Score={best_score[0]:.4f} | Time={elapsed:.2f}s")

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

print("\nExperimento BASELINE inaugural finalizado.")
print("Check:", output_csv)
print(f"Experiencias guardadas en ~/.autogoal/experience_store/{domain}/... (alias='{domain}')")