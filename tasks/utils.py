from moabb.datasets import Shin2017B, Shin2017A
from utils import extract_epochs, classify
from matplotlib import pyplot as plt


def single_param_cross_validator(config, label, ranges, apply_func):
    subjects = config['subjects']

    windows = config['windows']

    dataset = Shin2017B() if config['is_ma'] else Shin2017A()

    all_scores = []

    for subject in subjects:
        print("loading data for subject", subject)
        this_subject_data = dataset.get_data([subject])[subject]

        for this_value in ranges:
            print('using {} = {}'.format(label, this_value))

            apply_func(config, this_value)

            scores = []
            for window_start in windows:
                print('start at ', window_start, end=', ')
                data = extract_epochs(this_subject_data, config=config, start=window_start)
                score = classify(data, config=config)
                scores.append(score)
                print(score)
            all_scores.append(scores)

    # plot_scores(all_scores, config)
    windows = config['windows']
    for i, scores in enumerate(all_scores):
        plt.plot(windows, scores, label='Using {}={}'.format(label, ranges[i]))
        print(scores)
    plt.legend()
    plt.show()