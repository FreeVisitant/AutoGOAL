import numpy as np
from autogoal.meta_learning.feature_extraction._base import FeatureExtractor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

class MyTextClassificationFeatureExtractor(FeatureExtractor):
    """
    Extrae metafeatures de un dataset de texto:
      - n_instances, n_classes, entropy, imbalance, doc_length stats
      - landmarker accuracy (DecisionTree) tras SVD
    """
    def extract_features(self, X_train, y_train) -> np.ndarray:
        # Convert to numpy arrays
        X_train = np.array(X_train, dtype=object)
        y_train = np.array(y_train)

        n_instances = len(X_train)
        n_classes = len(np.unique(y_train))

        # Distribución de clases
        class_counts = np.array([np.sum(y_train == cls) for cls in np.unique(y_train)])
        class_probs = class_counts / n_instances
        class_entropy = -np.sum(class_probs * np.log2(class_probs + 1e-10))
        min_class_prob = np.min(class_probs)
        max_class_prob = np.max(class_probs)
        imbalance_ratio = (min_class_prob / max_class_prob) if max_class_prob > 0 else 0

        # Longitudes de docs
        doc_lengths = np.array([len(doc) for doc in X_train])
        avg_doc_length = float(np.mean(doc_lengths)) if doc_lengths.size else 0.0
        std_doc_length = float(np.std(doc_lengths)) if doc_lengths.size else 0.0
        coef_var_doc_length = (std_doc_length / avg_doc_length) if avg_doc_length != 0 else 0

        # Landmarker => Tfidf + SVD + DecisionTree
        vectorizer = TfidfVectorizer(max_features=5000, stop_words=set(stopwords.words('english')))
        X_vectorized = vectorizer.fit_transform(X_train)

        n_feat = X_vectorized.shape[1]
        n_comp = min(100, max(1, n_feat - 1))

        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_reduced = svd.fit_transform(X_vectorized)

        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_reduced, y_train, cv=skf)
        decision_tree_accuracy = np.mean(cv_scores)

        feature_vector = np.array([
            n_instances,
            n_classes,
            class_entropy,
            min_class_prob,
            max_class_prob,
            imbalance_ratio,
            avg_doc_length,
            std_doc_length,
            coef_var_doc_length,
            decision_tree_accuracy,
        ], dtype=float)

        return feature_vector
