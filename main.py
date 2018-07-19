from moabb1.datasets import Shin2017B
from utils import extract_epochs, classify
import logging
from matplotlib import pyplot as plt
import numpy as np

mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.ERROR)


def main():
    subjects = range(1, 11)

    dataset = Shin2017B()

    all_scores = []

    for subject in subjects:
        print("loading data for subject", subject)
        this_subject_data = dataset.get_data([subject])[subject]

        scores = []

        windows = range(0, 5)
        windows = range(-5, 20)
        for window_start in windows:
            print('start at ', window_start, end=', ')
            data = extract_epochs(this_subject_data, subject, start=window_start, duration=3.0)
            score = classify(data)
            scores.append(score)
            print(score)
        all_scores.append(scores)
    for scores in all_scores:
        plt.plot(scores, color='gray')

    scores = np.array(all_scores)
    print(scores, scores.shape)
    scores = np.average(scores, axis=0)
    print(scores, scores.shape)
    plt.plot(scores, linewidth=2)

    plt.show()



if __name__ == '__main__':
    main()
