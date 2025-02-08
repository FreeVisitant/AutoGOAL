import time
import csv
import os
import numpy as np
import random
import sys
import logging


from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.meta_learning._logging import ExperienceLogger
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn


from sklearn.feature_extraction.text import TfidfVectorizer


from autogoal.meta_learning.feature_extraction.text_classification import TextClassificationFeatureExtractor


from sklearn.datasets import fetch_20newsgroups

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


seeds = [0, 1]
search_iterations_list = [10, 50]

# Archivo CSV para almacenar resultados
output_csv = "experiment_20newsgroups_baseline.csv"
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

print("Cargando dataset 20 newsgroups...")


newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

X_text = newsgroups_train.data
y = newsgroups_train.target  
target_names = newsgroups_train.target_names

n_samples = len(X_text)
n_classes = len(set(y))

print(f"Documentos: {n_samples} | Número de clases: {n_classes}")
print("Etiquetas:", target_names)


# los textos son noticias, usamos ngram_range=(1,2) para capturar expresiones compuestas
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)
)
X_matrix = vectorizer.fit_transform(X_text).toarray()
n_features = X_matrix.shape[1]

print(f"TF-IDF shape: {X_matrix.shape}")



text_features_extractor = TextClassificationFeatureExtractor()
dataset_features = text_features_extractor.extract_features(X_text, y)


domain = "20newsgroups"


for seed in seeds:
    for search_iters in search_iterations_list:
        start_time = time.time()

        # Configurar ExperienceLogger con las meta-características extraídas
        experience_logger = ExperienceLogger(
            dataset_features=dataset_features,
            system_features=np.array([]),
            dataset_feature_extractor_name="TextClassificationFeatureExtractor",
            system_feature_extractor_name="NoSystem",
            alias=domain
        )

        # Crear el registry de algoritmos, excluyendo componentes innecesarios
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

print("\nExperimento BASELINE 20 newsgroups finalizado.")
print("Check:", output_csv)
print(f"Experiencias guardadas en ~/.autogoal/experience_store/{domain}/... (alias='{domain}')")
