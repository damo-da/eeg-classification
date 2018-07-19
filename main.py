from moabb1.datasets import Shin2017B, Shin2017A
from utils import extract_epochs, classify
import logging
from matplotlib import pyplot as plt
import numpy as np
from config import CONFIG

mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.ERROR)


def main():
    subjects = CONFIG['subjects']

    dataset = Shin2017B() if CONFIG['is_ma'] else Shin2017A()

    all_scores = []

    for subject in subjects:
        print("loading data for subject", subject)
        this_subject_data = dataset.get_data([subject])[subject]

        scores = []

        windows = CONFIG['windows']
        for window_start in windows:
            print('start at ', window_start, end=', ')
            data = extract_epochs(this_subject_data, subject, config=CONFIG, start=window_start)
            score = classify(data, config=CONFIG['classification'])
            scores.append(score)
            print(score)
        all_scores.append(scores)
    for scores in all_scores:
        plt.plot(scores, color='gray')

    scores = np.array(all_scores)
    scores = np.average(scores, axis=0)
    print(scores, scores.shape)
    plt.plot(scores, linewidth=2)

    plt.show()



if __name__ == '__main__':
    main()
