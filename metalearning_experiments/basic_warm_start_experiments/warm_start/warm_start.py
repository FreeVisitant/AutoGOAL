import math
import pprint
import numpy as np
from datetime import date
from typing import Dict, List, Optional, Union

from .experience import Experience, ExperienceStore
from .my_text_classification_extractor import MyTextClassificationFeatureExtractor

from autogoal.meta_learning.normalization import Normalizer
from autogoal.meta_learning.distance import DistanceMetric, EuclideanDistance, MahalanobisDistance
from autogoal.meta_learning.feature_extraction.system_feature_extractor import SystemFeatureExtractor
from autogoal.meta_learning import FeatureExtractor
from autogoal.sampling import update_model, UnormalizedWeightParam, ModelSampler

class WarmStart:
    """
    Ajusta el modelo probabilístico inicial de un AutoML
    basándose en experiencias pasadas y metafeatures.
    """
    def __init__(
        self,
        positive_min_threshold=0.2,
        k_pos=20,
        k_neg=20,
        max_alpha=0.05,
        min_alpha=-0.02,
        adaptative_positive_alpha_limit=None,
        adaptative_negative_alpha_limit=None,
        beta_scale=1.0,
        beta=None,
        f1_weight=0.5,
        evaluation_time_weight=0.5,
        normalizers: Optional[List[Normalizer]] = None,
        distance: DistanceMetric = EuclideanDistance,
        dataset_feature_extractor: Optional[FeatureExtractor] = MyTextClassificationFeatureExtractor,
        system_feature_extractor: Optional[FeatureExtractor] = SystemFeatureExtractor,
        from_date: Optional[Union[str, date]] = None, 
        to_date: Optional[Union[str, date]] = None,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
        exit_after_warmup: bool = False,
    ):
        self._model: Dict = {}
        self.generator_fn = None

        self.positive_min_threshold = positive_min_threshold
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha

        self.adaptative_positive_alpha_limit = adaptative_positive_alpha_limit
        self.adaptative_negative_alpha_limit = adaptative_negative_alpha_limit
        self.beta = beta
        self.beta_scale = beta_scale

        total_weight = f1_weight + evaluation_time_weight
        self.f1_weight = f1_weight / total_weight
        self.evaluation_time_weight = evaluation_time_weight / total_weight

        print(f"F1 Weight: {self.f1_weight}, Evaluation Time Weight: {self.evaluation_time_weight}")

        self.normalizers = normalizers or []
        self.distance = distance() if distance else EuclideanDistance()
        self.dataset_feature_extractor_class = dataset_feature_extractor
        self.system_feature_extractor_class = system_feature_extractor
        self.from_date = from_date
        self.to_date = to_date
        self.include = include
        self.exclude = exclude
        self.exit_after_warmup = exit_after_warmup

    def pre_warm_up(self, X_train, y_train):
        """
        Extrae metafeatures del dataset y sistema.
        """
        self.X_train = X_train
        self.y_train = y_train

        self.current_dataset_features = self._extract_meta_features(X_train, y_train)
        self.current_system_features = self._extract_system_features()

    def warm_up(self, generator_fn):
        """
        Ajusta el modelo inicial en base a experiencias pasadas.
        """
        self.generator_fn = generator_fn

        experiences = ExperienceStore.load_all_experiences(
            from_date=self.from_date, 
            to_date=self.to_date, 
            include=self.include, 
            exclude=self.exclude
        )

        experiences = self.filter_experiences_by_feature_extractors(experiences)
        if not experiences:
            print("No relevant experiences found. Skipping warm-up.")
            return

        distances = self.compute_distances(
            self.current_dataset_features, 
            self.current_system_features, 
            experiences
        )
        selected_pos, pos_dists, selected_neg, neg_dists = self.select_experiences(experiences, distances)

        if not selected_pos and not selected_neg:
            print("No experiences to adjust with. Skipping warm-up.")
            return

        alpha_experiences = self.compute_learning_rates(selected_pos, pos_dists, selected_neg, neg_dists)

        print(f"Learning from {len(selected_pos)} positive experiences and {len(selected_neg)} negative experiences.")
        self.adjust_model(alpha_experiences)

        if self.exit_after_warmup:
            raise ValueError("Exiting after warm-up (debug).")

        return self._model

    def compute_distances(self, current_dataset_features, current_system_features, experiences):
        combined_features = []
        for exp in experiences:
            combined = np.concatenate((exp.dataset_features, exp.system_features))
            combined_features.append(combined)

        current_combined = np.concatenate((current_dataset_features, current_system_features))
        combined_features.append(current_combined)

        normalized_combined_features = self._normalize_features(combined_features)

        # Actualizar cada exp con features normalizadas
        for i, exp in enumerate(experiences):
            combined = normalized_combined_features[i]
            dataset_len = len(exp.dataset_features)
            exp.dataset_features = combined[:dataset_len]
            exp.system_features = combined[dataset_len:]
        
        normalized_current_combined = normalized_combined_features[-1]
        self._prepare_distance_metric(normalized_combined_features)

        distances = []
        for exp in experiences:
            exp_feats = np.concatenate((exp.dataset_features, exp.system_features))
            dist = self.distance.compute(normalized_current_combined, exp_feats)
            distances.append(dist)

        return distances

    def select_experiences(self, experiences, distances):
        pairs = list(zip(experiences, distances))
        positive_pairs = [
            (exp, dist) for exp, dist in pairs
            if exp.f1 is not None
            and exp.f1 >= self.positive_min_threshold
            and exp.evaluation_time is not None 
            and exp.evaluation_time < np.Infinity
        ]
        negative_pairs = [
            (exp, dist) for exp, dist in pairs
            if exp.f1 is None
            or exp.f1 < self.positive_min_threshold
            or exp.f1 == -np.Infinity
            or exp.evaluation_time is None
            or exp.evaluation_time == np.Infinity
        ]

        sorted_pos = sorted(positive_pairs, key=lambda x: x[1])
        sorted_neg = sorted(negative_pairs, key=lambda x: x[1])

        selected_pos = sorted_pos[:self.k_pos] if self.k_pos else sorted_pos
        selected_neg = sorted_neg[:self.k_neg] if self.k_neg else sorted_neg

        pos_exps, pos_dists = zip(*selected_pos) if selected_pos else ([], [])
        neg_exps, neg_dists = zip(*selected_neg) if selected_neg else ([], [])

        return list(pos_exps), list(pos_dists), list(neg_exps), list(neg_dists)

    def compute_learning_rates(self, selected_positive_exps, positive_distances, selected_negative_exps, negative_distances):
        experience_alphas = {}

        if self.adaptative_negative_alpha_limit and len(selected_negative_exps) > 0:
            self.min_alpha = self.adaptative_negative_alpha_limit / len(selected_negative_exps)

        if self.adaptative_positive_alpha_limit and len(selected_positive_exps) > 0:
            self.max_alpha = self.adaptative_positive_alpha_limit / len(selected_positive_exps)

        all_exps = selected_positive_exps + selected_negative_exps
        all_dists = positive_distances + negative_distances

        beta = self.beta if self.beta is not None else self.compute_distance_decay_beta(all_dists)
        exp_dist_map = dict(zip(all_exps, all_dists))

        # Positivas
        if selected_positive_exps:
            # Agrupamos por alias
            group_map = {}
            for exp in selected_positive_exps:
                alias = exp.alias or "Unknown"
                group_map.setdefault(alias, []).append(exp)

            for alias, group_list in group_map.items():
                f1s = [e.f1 for e in group_list]
                times = [e.evaluation_time for e in group_list]
                max_f1 = max(f1s) if f1s else 1.0
                min_time = min(times) if times else 1.0
                max_time = max(times) if times else 1.0
                time_range = max_time - min_time if max_time != min_time else 1.0

                for exp in group_list:
                    distance = exp_dist_map[exp]
                    norm_f1 = exp.f1 / max_f1
                    norm_time = (exp.evaluation_time - min_time)/time_range if time_range > 0 else 0
                    norm_time_inv = 1 - norm_time

                    utility = self.f1_weight * norm_f1 + self.evaluation_time_weight * norm_time_inv
                    weight = math.exp(-beta * distance)
                    alpha = self.max_alpha * utility * weight
                    alpha = min(alpha, self.max_alpha)
                    experience_alphas[exp] = alpha

        if selected_negative_exps:
            for exp in selected_negative_exps:
                distance = exp_dist_map[exp]
                weight = math.exp(-beta * distance)
                alpha = self.min_alpha * weight
                alpha = max(alpha, self.min_alpha)
                experience_alphas[exp] = alpha

        return experience_alphas

    def compute_distance_decay_beta(self, all_distances):
        if not all_distances:
            return 0.0
        mean_d = np.mean(all_distances)
        std_d = np.std(all_distances)
        return self.beta_scale / (std_d + 1e-8)

    def adjust_model(self, alpha_experiences):
        from autogoal.sampling import update_model, UnormalizedWeightParam, ModelSampler

        exp_alpha_pairs = sorted(alpha_experiences.items(), key=lambda x: x[1], reverse=True)

        for exp, alpha in exp_alpha_pairs:
            sampler = ModelSampler(self._model)
            self.generator_fn(sampler)
            self._model = update_model(self._model, sampler.updates, alpha)

            for item, value in self._model.items():
                if isinstance(value, UnormalizedWeightParam) and value.value == 0:
                    self._model[item] = UnormalizedWeightParam(value=0.001)

    def filter_experiences_by_feature_extractors(self, experiences):
        ds_name = self.dataset_feature_extractor_class.__name__
        sys_name = self.system_feature_extractor_class.__name__

        return [
            exp for exp in experiences
            if exp.dataset_feature_extractor_name == ds_name
               and exp.system_feature_extractor_name == sys_name
        ]

    def _normalize_features(self, feature_vectors_list):
        if not feature_vectors_list:
            return []
        feature_matrix = np.vstack(feature_vectors_list)
        for normalizer in self.normalizers:
            feature_matrix = normalizer.fit_transform(feature_matrix)
        num_vecs = len(feature_vectors_list)
        splitted = np.vsplit(feature_matrix, num_vecs)
        splitted = [v.flatten() for v in splitted]
        return splitted

    def _extract_meta_features(self, X_train, y_train):
        extractor = self.dataset_feature_extractor_class()
        return extractor.extract_features(X_train, y_train)

    def _extract_system_features(self):
        extractor = self.system_feature_extractor_class()
        return extractor.extract_features()

    def _prepare_distance_metric(self, combined_features):
        if isinstance(self.distance, MahalanobisDistance):
            covariance = np.cov(combined_features, rowvar=False)
            try:
                VI = np.linalg.inv(covariance)
            except np.linalg.LinAlgError:
                reg = 1e-6 * np.eye(covariance.shape[0])
                VI = np.linalg.inv(covariance + reg)
            self.distance.set_VI(VI)
