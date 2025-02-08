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


import nltk
try:
    nltk.data.find('corpora/names')
    logging.info("El corpus 'names' ya está descargado.")
except LookupError:
    logging.info("Descargando el corpus 'names' de NLTK...")
    nltk.download('names')

from nltk.corpus import names


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

output_csv = "experiment_nltk_names_baseline.csv"
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

print("Cargando corpus 'names' de NLTK...")


# Extraemos los nombres masculinos y femeninos y creamos ejemplos (texto, etiqueta)
male_names = names.words('male.txt')
female_names = names.words('female.txt')

documents = []
for n in male_names:
    documents.append((n.lower(), 'male'))
for n in female_names:
    documents.append((n.lower(), 'female'))

# Barajamos los documentos para evitar sesgos de orden
random.shuffle(documents)
X_all, y_all = zip(*documents)
X_all = list(X_all)
y_all = list(y_all)


X_text = X_all
y = y_all

n_samples = len(X_text)
n_classes = len(set(y))

print(f"Documentos: {n_samples} | n_classes: {n_classes}")
print("Etiquetas:", list(set(y)))



# Se utiliza "char_wb" para extraer n-grams de caracteres que ocurren dentro de las palabras.
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 4),
    max_features=1000
)
X_matrix = vectorizer.fit_transform(X_text).toarray()
n_features = X_matrix.shape[1]

print(f"TF-IDF shape: {X_matrix.shape}")


text_features_extractor = TextClassificationFeatureExtractor()
dataset_features = text_features_extractor.extract_features(X_text, y)


domain = "nltk_names"

for seed in seeds:
    for search_iters in search_iterations_list:
        start_time = time.time()

        # Creamos el logger utilizando la nueva subclase que registra evaluation time real
        experience_logger = ExperienceLoggerWithEvalTime(
            dataset_features=dataset_features,
            system_features=np.array([]),
            dataset_feature_extractor_name="TextClassificationFeatureExtractor",
            system_feature_extractor_name="NoSystem",
            alias=domain
        )

        # Configuramos el registry de algoritmos (excluyendo vectorizadores)
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

        best_score = automl.best_scores_[0]  # Por ejemplo, la exactitud
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

print("\nExperimento BASELINE nltk_names finalizado.")
print("Check:", output_csv)
print(f"Experiencias guardadas en ~/.autogoal/experience_store/{domain}/... (alias='{domain}')")
