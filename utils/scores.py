from matplotlib import pyplot as plt
import numpy as np


def plot_scores(all_scores, config):
    windows = config['windows']
    for scores in all_scores:
        plt.plot(windows, scores, color='gray')

    scores = np.array(all_scores)

    scores = np.average(scores, axis=0)
    print(scores)
    plt.plot(windows, scores, linewidth=2)

    plt.show()
