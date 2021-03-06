from utils import extract_epochs, classify, plot_scores, apply_algorithm, get_window


def by_window_func(subject, config, dataset):
    print("loading data for subject", subject)

    this_subject_data = dataset.get_data([subject])[subject]
    this_subject_data = extract_epochs(this_subject_data, config)

    scores = []

    windows = config['windows']

    for window_start in windows:
        data = get_window(this_subject_data, config=config, start=window_start)
        score = classify(data, config=config)
        print(score)
        scores.append(score)

    return scores


def by_window(config):
    result = apply_algorithm(by_window_func, config)
    plot_scores(result, config)
