import random
import csv

from autogoal.datasets import download, datapath

def load(*args, **kwargs):
    try:
        download("dbpedia")
    except Exception as e:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    path = datapath("dbpedia")

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    classes = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]
    with open(path / "train.csv", "r") as fd:
        reader = csv.reader(fd)
        for row in reader:
            X_train.append({
                "title": row[1],
                "content": row[2]
            })
            y_train.append(classes[int(row[0])])
            
    with open(path / "test.csv", "r") as fd:
        reader = csv.reader(fd)
        for row in reader:
            X_test.append({
                "title": row[1],
                "content": row[2]
            })
            y_test.append(classes[int(row[0])])
            
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    load()