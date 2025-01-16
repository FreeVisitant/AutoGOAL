import os
import datetime
import pickle
import numpy as np

class Experience:
    """
    Representa una 'experiencia' de entrenamiento:
    - dataset_features, system_features, f1, evaluation_time, etc.
    - algorithms (pipeline info, opcional).
    """
    def __init__(
        self,
        dataset_feature_extractor_name,
        system_feature_extractor_name,
        dataset_features,
        system_features,
        f1=None,
        evaluation_time=None,
        alias=None,
        date=None,
        algorithms=None,
    ):
        self.dataset_feature_extractor_name = dataset_feature_extractor_name
        self.system_feature_extractor_name = system_feature_extractor_name
        self.dataset_features = dataset_features
        self.system_features = system_features
        self.f1 = f1
        self.evaluation_time = evaluation_time
        self.alias = alias
        self.date = date if date else datetime.datetime.now()

        if algorithms is None:
            algorithms = []
        self.algorithms = algorithms

    def __repr__(self):
        return f"<Experience alias={self.alias}, f1={self.f1}, time={self.evaluation_time}>"

class ExperienceStore:
    """
    Carga/guarda experiencias en ficheros .pkl.
    Ajusta la lógica si quieres filtrar por alias, fechas, etc.
    """

    @staticmethod
    def load_all_experiences(from_date=None, to_date=None, include=None, exclude=None, base_dir="experiences_logs"):
        if not os.path.exists(base_dir):
            print(f"No experience directory found at {base_dir}. Returning empty list.")
            return []

        experiences = []
        for fname in os.listdir(base_dir):
            if not fname.endswith(".pkl"):
                continue
            fpath = os.path.join(base_dir, fname)

            with open(fpath, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, Experience):
                    data = [data]
                elif not isinstance(data, list):
                    data = [data]

                for exp in data:
                    if not isinstance(exp, Experience):
                        continue

                    # Filtrar por alias
                    if include and (include not in (exp.alias or "")):
                        continue
                    if exclude and (exclude in (exp.alias or "")):
                        continue

                    # Filtrar por fecha
                    if from_date and exp.date < parse_date(from_date):
                        continue
                    if to_date and exp.date > parse_date(to_date):
                        continue

                    experiences.append(exp)

        return experiences

def parse_date(value):
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    return None
