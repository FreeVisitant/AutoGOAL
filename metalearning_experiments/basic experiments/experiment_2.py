import csv, os
import time
from sklearn.datasets import load_iris, load_wine
from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn

datasets = {
    "iris": load_iris,
    "wine": load_wine
}

seeds = [0, 1, 2]
iters_list = [10, 50]

output_csv = "experimentos.csv"
first_time = not os.path.exists(output_csv)

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)
    if first_time:
        writer.writerow([
            "dataset", "random_state", "search_iterations",
            "score", "pipeline", "time_seconds"
        ])

    for dataset_name, loader in datasets.items():
        X, y = loader(return_X_y=True)

        for seed in seeds:
            for iters in iters_list:
                start_t = time.time()

                # Preparar registry
                my_registry = find_classes(modules=[sklearn])

                automl = AutoML(
                    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
                    output=VectorCategorical,
                    registry=my_registry,
                    random_state=seed,
                    search_iterations=iters,
                )

                automl.fit(X, y)

                elapsed = time.time() - start_t

                # Normalmente, el 1er pipeline es el de mayor score
                best_score = automl.best_scores_[0]
                best_pipeline = str(automl.best_pipelines_[0])

                writer.writerow([
                    dataset_name, seed, iters,
                    best_score[0],  # score[0] si es una tupla
                    best_pipeline, f"{elapsed:.2f}"
                ])

                print(f"{dataset_name} - seed={seed}, iters={iters} -> {best_score[0]} in {elapsed:.2f}s")
