import random
import csv

from autogoal.datasets import download, datapath

def load(*args, **kwargs):
    try:
        download("yelp_reviews")
    except Exception as e:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    path = datapath("yelp_reviews")

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    with open(path / "train.csv", "r") as fd:
        reader = csv.reader(fd)
        title_line = True
        for row in reader:
            if title_line:
                title_line = False
                continue
            X_train.append(row[1])
            y_train.append(int(row[0]))
            
    with open(path / "test.csv", "r") as fd:
        reader = csv.reader(fd)
        title_line = True
        for row in reader:
            if title_line:
                title_line = False
                continue
            X_test.append(row[1])
            y_test.append(int(row[0]))
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    load()