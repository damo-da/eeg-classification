from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from mne.decoding import CSP
import numpy as np


def classify(epochs, n_splits=10, test_size=0.2):
    classifier = linear_model.LogisticRegression()
    csp = CSP(norm_trace=False)

    labels = epochs.events[:, -1]

    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)
    scores = []
    epochs_data = epochs.get_data()

    for train_idx, test_idx in cv.split(labels):
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train = csp.fit_transform(epochs_data[train_idx], y_train)
        x_test = csp.transform(epochs_data[test_idx])

        classifier.fit(x_train, y_train)

        scores.append(classifier.score(x_test, y_test))

    return np.mean(scores)