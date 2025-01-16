import os
import csv
import time

from autogoal.datasets import (
    ag_news,
    imdb_50k_movie_reviews,
    yelp_reviews,
    rotten_tomatoes,
)
from autogoal.ml import AutoML
from autogoal.meta_learning._logging import ExperienceLogger
from autogoal.meta_learning.feature_extraction import TextClassificationFeatureExtractor


from my_final_experiments.my_tfidf_bridge import MyTfidfVectorizerBridge

from autogoal_sklearn._generated import MultinomialNB, LogisticRegression, SVC

from autogoal.kb import Seq, Sentence, Supervised, VectorCategorical

from autogoal.search import ConsoleLogger, RichLogger

def load_dataset(dataset, max_examples=1000):
    # dataset.load(split=True) => X_train, y_train, X_test, y_test
    X_train, y_train, X_test, y_test = dataset.load(split=True)
    X_train, y_train = X_train[:max_examples], y_train[:max_examples]
    X_test, y_test = X_test[:max_examples], y_test[:max_examples]
    return X_train, y_train, X_test, y_test

DATASETS = {
    "AGNews": ag_news,
    "IMDB": imdb_50k_movie_reviews,
    "YelpReviews": yelp_reviews,
    "RottenTomatoes": rotten_tomatoes,
}

def run_baseline_experiment(
    alias: str,
    dataset,
    output_csv="baseline_text_results.csv",
    max_examples=1000,
    search_iterations=50,
):
    print(f"\n=== Baseline experiment: {alias} ===")
    X_train, y_train, X_test, y_test = load_dataset(dataset, max_examples=max_examples)

    feat_ext = TextClassificationFeatureExtractor()
    meta_vector = feat_ext.extract_features(X_train, y_train)
    print(f"Metafeature vector for {alias}:", meta_vector)

    logger = ExperienceLogger(
        dataset_features=meta_vector,
        system_features=None,
        dataset_feature_extractor_name="TextClassificationFeatureExtractor",
        system_feature_extractor_name="SystemFeatureExtractor",
        alias=alias,
    )

    registry = [
        MyTfidfVectorizerBridge, 
        MultinomialNB,
        LogisticRegression,
        SVC,
    ]

    # input=(Seq[Sentence], Supervised[VectorCategorical]) => output=VectorCategorical
    automl = AutoML(
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=registry,
        search_iterations=search_iterations,
        cross_validation_steps=1,
        errors="warn",
        search_timeout=120,      # definir min
        evaluation_timeout=20,   
    )

    search_loggers = [ConsoleLogger(), RichLogger(), logger]

    start = time.time()
    automl.fit(X_train, y_train, logger=search_loggers)
    train_time = time.time() - start

    best_score = None
    if len(X_test) > 0 and len(y_test) > 0:
        scores = automl.score(X_test, y_test)
        if scores:
            best_score = max(s[0] for s in scores)

    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "alias","n_train","n_test","search_iterations","best_score","train_time_sec"
            ])

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            alias,
            len(X_train),
            len(X_test),
            search_iterations,
            best_score,
            round(train_time,2)
        ])

    print(f"[{alias}] Baseline done. best_score={best_score}, time={round(train_time,2)}s")

def main():
    output_csv = "baseline_text_results.csv"
    if os.path.exists(output_csv):
        os.remove(output_csv)

    for alias, ds in DATASETS.items():
        run_baseline_experiment(
            alias=alias,
            dataset=ds,
            output_csv=output_csv,
            max_examples=1000,
            search_iterations=50,
        )

if __name__=="__main__":
    main()
