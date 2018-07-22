from utils import extract_epochs, classify


def by_window(subject, config, dataset):
    print("loading data for subject", subject)

    this_subject_data = dataset.get_data([subject])[subject]

    scores = []

    windows = config['windows']

    for window_start in windows:
        data = extract_epochs(this_subject_data, config=config, start=window_start)
        score = classify(data, config=config)
        print(score)
        scores.append(score)

    return scores


