from moabb.datasets import Shin2017A, Shin2017B
from utils import extract_epochs, classify
from utils.scores import plot_scores


def by_window(config):
    subjects = config['subjects']

    dataset = Shin2017B() if config['is_ma'] else Shin2017A()

    all_scores = []

    for subject in subjects:
        print("loading data for subject", subject)
        this_subject_data = dataset.get_data([subject])[subject]

        scores = []

        windows = config['windows']
        for window_start in windows:
            print('start at ', window_start, end=', ')
            data = extract_epochs(this_subject_data, subject, config=config, start=window_start)
            score = classify(data, config=config)
            scores.append(score)
            print(score)
        all_scores.append(scores)

    plot_scores(all_scores, config)
