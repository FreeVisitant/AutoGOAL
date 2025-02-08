import time
import csv
import os
import numpy as np

from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.meta_learning._logging import ExperienceLogger
from autogoal.meta_learning.warm_start import WarmStart
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn
from autogoal.search._warm_start_pge import NSPEWarmStartSearch
from autogoal.datasets import imdb_50k_movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from autogoal.meta_learning.feature_extraction.text_classification import TextClassificationFeatureExtractor


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

output_csv = "experiment_imdb_warm_start.csv"
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

print("Cargando IMDB 50k Movie Reviews...")
X_train, y_train, X_test, y_test = imdb_50k_movie_reviews.load()


X_text = X_train
y = y_train

n_samples = len(X_text)
n_classes = len(set(y))


vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=3000
)
X_matrix = vectorizer.fit_transform(X_text).toarray()
n_features = X_matrix.shape[1]

print(f"Documentos: {n_samples} | TF-IDF shape: {X_matrix.shape} | n_classes={n_classes}")

# Extracción de metafeatures usando TextClassificationFeatureExtractor
text_features_extractor = TextClassificationFeatureExtractor()
dataset_features = text_features_extractor.extract_features(X_train, y_train)

domain = "imdb_warm"


warm_start = WarmStart(
    positive_min_threshold=0.2,
    k_pos=20,
    k_neg=0,
    max_alpha=0.05,
    min_alpha=-0.02,
    beta_scale=1.0,
    normalizers=[],
    distance=None,  # Usará EuclideanDistance por defecto
    dataset_feature_extractor=TextClassificationFeatureExtractor,
    system_feature_extractor=None,
    from_date=None,
    to_date=None,
    include=".*",  # Carga todas las experiencias
    exclude="^(imdb_warm|imdb)$",  # Excluye la experiencia actual y la del baseline de imdb
    exit_after_warmup=False
)



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
            search_iterations=search_iters,
            search_algorithm=NSPEWarmStartSearch,
            warm_start=warm_start
        )

        # Pre-calcula las metafeatures del dataset actual para el warm start
        warm_start.pre_warm_up(X_train, y_train)

        automl.fit(X_matrix, y, logger=[experience_logger])
        elapsed = time.time() - start_time

        best_score = automl.best_scores_[0]
        best_pipeline = automl.best_pipelines_[0]

        print(f"[IMDB WARM_START] SEED={seed} | ITERS={search_iters} -> Score={best_score[0]:.4f} | Time={elapsed:.2f}s")

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

print("\nExperimento WARM_START IMDB finalizado.")
print("Check:", output_csv)
print(f"Experiencias guardadas en ~/.autogoal/experience_store/{domain}/... (alias='{domain}')")