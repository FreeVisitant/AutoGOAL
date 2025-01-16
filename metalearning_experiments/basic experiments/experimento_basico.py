from autogoal.datasets import cars
from autogoal.kb import (MatrixContinuousDense, Supervised, VectorCategorical)
from autogoal.ml import AutoML

X, y = cars.load()

automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical
)

# Entrenar el modelo automáticamente
automl.fit(X, y)

# Mostrar el mejor pipeline encontrado y su puntaje
print("Mejor pipeline encontrado:", automl.best_pipeline_)
print("Mejor puntaje:", automl.best_score_)
