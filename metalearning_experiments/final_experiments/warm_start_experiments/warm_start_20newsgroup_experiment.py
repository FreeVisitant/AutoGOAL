import monkey_patch_numpy_warnings

import time
import csv
import os
import numpy as np

from autogoal.search import RichLogger
from autogoal.ml import AutoML
from autogoal.meta_learning import WarmStart
from autogoal.meta_learning._logging import ExperienceLogger
from autogoal.meta_learning.feature_extraction.text_classification import TextClassificationFeatureExtractor
from autogoal.search._warm_start_pge import NSPEWarmStartSearch

from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


from autogoal.meta_learning.feature_extraction._base import FeatureExtractor

class NoSystemFeatureExtractor(FeatureExtractor):
    def extract_features(self, *args, **kwargs) -> np.ndarray:
        # Devuelve un array vacío (0-length)
        return np.array([], dtype=np.float32)


seeds = [0, 1]
search_iterations_list = [10, 50]
output_csv = "experiment_20newsgroups_warm.csv"

first_time = not os.path.exists(output_csv)
if first_time:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "warm_alias",
            "n_samples",
            "n_features",
            "n_classes",
            "seed",
            "search_iters",
            "best_score",
            "time_seconds",
            "best_pipeline"
        ])

categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball']
print("Cargando datos de 20newsgroups...")
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
X_text = newsgroups.data  # Lista de textos
y = newsgroups.target     # Etiquetas numéricas
print("Total documentos:", len(X_text))


random_state_split = 42
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=random_state_split
)
print("Documentos entrenamiento:", len(X_train_text))
print("Documentos validación:", len(X_val_text))


vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X_train_matrix = vectorizer.fit_transform(X_train_text).toarray()
X_val_matrix = vectorizer.transform(X_val_text).toarray()

print("Matriz TF-IDF entrenamiento:", X_train_matrix.shape)
print("Matriz TF-IDF validación:", X_val_matrix.shape)

n_samples = len(X_text)
n_features = X_train_matrix.shape[1]
n_classes = len(np.unique(y))


dataset_extractor = TextClassificationFeatureExtractor()
dataset_features = dataset_extractor.extract_features(X_text, y)


warm_alias = "20newsgroups_warm"

warm = WarmStart(
    positive_min_threshold=0.2,
    k_pos=20,
    k_neg=20,
    max_alpha=0.05,
    min_alpha=-0.02,
    dataset_feature_extractor=TextClassificationFeatureExtractor,
    system_feature_extractor=NoSystemFeatureExtractor,
    include=".*",
    exclude=f"({warm_alias})"
)

# Pre-warm: extraemos las metafeatures del dataset actual usando los textos de entrenamiento y sus etiquetas
warm.pre_warm_up(X_train_text, y_train)

def generator_fn(sampler):

    pass

print(f"[WARM] Cargando experiences, excluyendo alias={warm_alias}")
warm.warm_up(generator_fn)


exclude_pattern = (
    "("
    "CountVectorizer|TfidfVectorizer|HashingVectorizer|"
    "KernelPCA|KernelCenterer|AdditiveChi2Sampler|Nystroem|"
    "AffinityPropagation|Isomap|MeanShift|PolynomialFeatures|"
    "RadiusNeighborsRegressor|RadiusNeighbors Transformer|"
    "SVC|SVR|"
    "KBinsDiscretizer|DecisionTreeClassifier|PowerTransformer|RobustScaler|FastICA|FeatureAgglomeration|KNNImputer|"
    "LatentDirichletAllocation|LocalOutlierFactor|NMF|OneClassSVM|"
    "OrthogonalMatchingPursuit|RadiusNeighborsTransformer|PCA|SGDOneClassSVM|"
    "AggregatedTransformer|"
    "Birch|"
    "FactorAnalysis|"
    "FeatureDenseVectorizer|"
    "KMeans|"
    "KNeighborsRegressor|"
    "KNeighborsTransformer|"
    "QuantileRegressor|"
    "TheilSenRegressor|"
    "CRFTagger|"
    "ClassifierTagger|"
    "ClassifierTransformerTagger|"
    "SparseClassifierTagger|"
    "SparseClassifierTransformerTagger|"
    "ARDRegression|BayesianRidge|KNeighborsClassifier|RidgeClassifier|DecisionTreeRegressor|ElasticNet|GammaRegressor|GeneralizedLinearRegressor|"
    "HuberRegressor|Lars|Lasso|LassoLars|LassoLarsIC|LinearRegression|PassiveAggressiveRegressor|"
    "PoissonRegressor|SGDRegressor|TruncatedSVD|SplineTransformer|TweedieRegressor|MinMaxScaler|StandardScaler|LabelBinarizer|"
    "FeatureSparseVectorizer|SparseAggregatedVectorizer"
    ")"
)
my_registry = find_classes(modules=[sklearn], exclude=exclude_pattern)


experience_logger = ExperienceLogger(
    alias=warm_alias,
    dataset_features=dataset_features,
    system_features=np.array([]),
    dataset_feature_extractor_name="TextClassificationFeatureExtractor",
    system_feature_extractor_name="NoSystem",
)

for seed in seeds:
    for search_iters in search_iterations_list:
        start_time = time.time()

        automl = AutoML(
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=my_registry,
            random_state=seed,
            search_iterations=search_iters,
            errors="warns"  
        )

        # Cambiar el search_algorithm a NSPEWarmStartSearch
        from autogoal.search._warm_start_pge import NSPEWarmStartSearch
        automl.search_algorithm = NSPEWarmStartSearch


        automl.search_kwargs["warm_start"] = warm

        # Entrenamiento (fit) con logger
        automl.fit(X_train_matrix, y_train, logger=[experience_logger])
        elapsed = time.time() - start_time

        best_score = automl.best_scores_[0]
        best_pipeline = automl.best_pipelines_[0]

        print(f"[20NG_WARM] SEED={seed}, ITERS={search_iters} => Score={best_score[0]:.4f}, Time={elapsed:.2f}s")

        # Guardar resultados en CSV
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                warm_alias,
                n_samples,
                n_features,
                n_classes,
                seed,
                search_iters,
                best_score[0],
                f"{elapsed:.2f}",
                str(best_pipeline)
            ])

print("\n[WARM] Experimento 20newsgroups finalizado.")
print(f"Revisa CSV: {output_csv}")
print(f"Experiencias guardadas en ~/.autogoal/experience_store/{warm_alias}/ ...")
