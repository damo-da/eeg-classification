from utils import extract_epochs, classify, get_window


def single_param_cross_validator_func(subject, config, dataset, label, values, apply_func):
    windows = config['windows']

    all_scores = []

    full_data = dataset.get_data([subject])[subject]

    for this_value in values:
        print('using {} = {}'.format(label, this_value))

        apply_func(config, this_value)

        print('extracting epoch for subject ', subject)
        this_subject_data = extract_epochs(full_data, config)
        print('extraction complete for ', subject)

        scores = []
        for window_start in windows:
            print('start at ', window_start, end=', ')
            data = get_window(this_subject_data, config=config, start=window_start)
            score = classify(data, config=config)
            scores.append(score)
            print(score)
        all_scores.append(scores)

    return all_scores
