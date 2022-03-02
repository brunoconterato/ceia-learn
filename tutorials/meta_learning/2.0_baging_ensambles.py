from numpy import mean, std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

N = 10

X, y = make_classification()

models = []

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=36851234)

for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    model = MLPClassifier()

    print(f"Training the module number: {i+1}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    model_score = cross_val_score(model, X_test, y_test)
    print(f"Training model number {i+1} score: {model_score}")
    models.append(model)

ensambler = MLPClassifier()