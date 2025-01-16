import numpy as np
from sklearn.datasets import load_iris

from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.contrib.sklearn import find_classes

X, y = load_iris(return_X_y=True)

automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_iterations=10,
    random_state=42,
    registry=find_classes(),  
)

automl.fit(X, y)

print("Best pipeline:", automl.best_pipeline_)
print("Best score:", automl.best_score_)
