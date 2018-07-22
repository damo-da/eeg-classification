from matplotlib import pyplot as plt
import numpy as np


def plot_scores(all_scores, config):
    windows = config['windows']
    for scores in all_scores:
        plt.plot(windows, scores, color='gray')

    scores = np.array(all_scores)

    scores = np.average(scores, axis=0)
    plt.plot(windows, scores, linewidth=2)

    plt.show()


def plot_cv_scores(all_scores, label, values, config):
    assert (len(all_scores) > 0)

    windows = config['windows']

    all_scores = np.array(all_scores)

    averaged_scores = np.mean(all_scores, axis=0)  # average between subjects

    plt.title('Cross validating {}'.format(label))
    for i, value in enumerate(values):
        plt.plot(windows, averaged_scores[i, :], label='value {}'.format(value))

    plt.legend()

    plt.show()
