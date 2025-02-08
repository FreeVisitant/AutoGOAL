import numpy as np
import pandas as pd
from autogoal.datasets import ag_news, imdb_50k_movie_reviews, rotten_tomatoes, yelp_reviews
from autogoal.meta_learning import (
    TextClassificationFeatureExtractor,
    LogNormalizer,
    MinMaxNormalizer,
    EuclideanDistance
)

###############################################################################
# Creación de la clase "MinMaxLogNormalizer" combinando Log + MinMax
###############################################################################
class MinMaxLogNormalizer:
    """
    Aplica primero log-transform y luego min-max scaling.
    Parametros:
    log_epsilon: float para el log
    minmax_epsilon: float para el minmax
    """
    def __init__(self, log_epsilon: float = 1e-8, minmax_epsilon: float = 1e-8):
        self.log_epsilon = log_epsilon
        self.minmax_epsilon = minmax_epsilon
        self.log_ = LogNormalizer(epsilon=self.log_epsilon)
        self.minmax_ = MinMaxNormalizer(epsilon=self.minmax_epsilon)

    def fit(self, feature_vectors: np.ndarray):
        # 1) Aplico log y luego ajusto minmax
        logged = self.log_.fit_transform(feature_vectors)
        self.minmax_.fit(logged)

    def transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        logged = self.log_.transform(feature_vectors)
        return self.minmax_.transform(logged)

    def fit_transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        self.fit(feature_vectors)
        return self.transform(feature_vectors)

###############################################################################
# Funciones del script principal
###############################################################################

def load_dataset(dataset):
    X_train, y_train, X_test, y_test = dataset.load(True)
    return X_train, y_train, X_test, y_test

def main():
    datasets = {
        'IMDB': imdb_50k_movie_reviews,
        'YelpReviews': yelp_reviews,
    }
    
    feature_extractor = TextClassificationFeatureExtractor()
    feature_vectors = []
    dataset_names = []
    
    # Extrae features con solo X_train, y_train (IGNORAMOS X_test, y_test)
    for name, dataset in datasets.items():
        print(f'Processing dataset: {name}')
        X_train, y_train, _, _ = load_dataset(dataset)
        feature_vector = feature_extractor.extract_features(X_train, y_train)
        feature_vectors.append(feature_vector)
        dataset_names.append(name)
    
  
    # Convert feature vectors to numpy array
    feature_vectors = np.array(feature_vectors)
    
    # Normalize all feature vectors together using MinMaxLogNormalizer
    normalizer = MinMaxLogNormalizer(log_epsilon=1e-8, minmax_epsilon=1e-8)
    normalizer.fit(feature_vectors)
    normalized_vectors = normalizer.transform(feature_vectors)
    
    # Compute distances
    distance_metric = EuclideanDistance()
    distance_matrix = distance_metric.compute_pairwise(normalized_vectors)

    
    # Display the results
    print('\n===== Normalized Feature Vectors =====')
    for name, vec in zip(dataset_names, normalized_vectors):
        print(f'{name}: {vec}')
    
    print('\n===== Pairwise Dataset Distances (Euclidean) =====')
    distance_df = pd.DataFrame(distance_matrix, index=dataset_names, columns=dataset_names)
    print(distance_df)

if __name__ == '__main__':
    main()