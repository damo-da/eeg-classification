from sklearn import linear_model
from sklearn.model_selection import RepeatedKFold
from mne.decoding import CSP
import numpy as np


def classify(epochs, config):
    n_splits = config['classification']['n_splits']
    n_repeats = config['classification']['n_repeats']
    classifier = config['classification']['classifier']
    csp = CSP(n_components=config['classification']['csp_num_components'],
              norm_trace=config['classification']['csp_norm_trace'])

    labels = epochs.events[:, -1]

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    scores = []
    epochs_data = epochs.get_data()

    for train_idx, test_idx in cv.split(labels):
        y_train, y_test = labels[train_idx], labels[test_idx]

        x_train = csp.fit_transform(epochs_data[train_idx], y_train)
        x_test = csp.transform(epochs_data[test_idx])

        classifier.fit(x_train, y_train)

        scores.append(classifier.score(x_test, y_test))

    return np.mean(scores)
