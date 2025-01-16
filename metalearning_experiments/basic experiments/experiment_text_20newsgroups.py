import time
import csv
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal_contrib import find_classes
import autogoal_sklearn as sklearn


categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball']
search_iterations = [10, 50]  
seeds = [0, 1]  

output_csv = "experiment_text_20newsgroups.csv"
first_time = not os.path.exists(output_csv)

with open(output_csv, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if first_time:
        writer.writerow([
            "categories", "random_state", "search_iters",
            "best_score", "time_seconds", "best_pipeline"
        ])

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
X_text = newsgroups_train.data  
y = newsgroups_train.target    

vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X_matrix = vectorizer.fit_transform(X_text).toarray()
print("Documentos de texto:", len(X_text), "-> Matriz TF-IDF:", X_matrix.shape)
for seed in seeds:
    for iters in search_iterations:
        start_time = time.time()
        my_registry = find_classes(modules=[sklearn])
        automl = AutoML(
            input=(MatrixContinuousDense, Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=my_registry,
            random_state=seed,
            search_iterations=iters,
        )
        automl.fit(X_matrix, y)

        elapsed = time.time() - start_time
        best_score = automl.best_scores_[0]  
        best_pipeline = automl.best_pipelines_[0]
        print(f"Categories={categories}, Seed={seed}, Iters={iters} -> Score={best_score[0]:.4f}, Time={elapsed:.2f}s")
        
        with open(output_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                categories, seed, iters,
                best_score[0],
                f"{elapsed:.2f}",
                str(best_pipeline)
            ])

print("\nExperimentos finalizados. Revisa el archivo:", output_csv)
