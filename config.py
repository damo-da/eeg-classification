from sklearn.svm import SVC
import random
import numpy as np

config = {
    'is_ma': True,
    'fps': 200,

    # 'concurrency': True,
    'concurrency': False,

    'random_state': 100,
    # 'random_state': None,

    # 'subjects': range(1, 10),
    'subjects': range(1, 2),

    # 'windows': range(-5, 20),
    'windows': range(-5, 10),
    'window_duration': 3.0,

    'epoch_start': -5,
    'epoch_end': 25,

    'baseline': (-3.0, 0-0.05),

    'filter': {
        'ma': {
            'Wp': (0.0400, 0.3500),  # passband
            'Ws': (0.0100, 0.3800),  # stopband
            'Rp': 3,
            'Rs': 30,
        },
        'mi': {
            'Wp': (0.0950, 0.1450),
            'Ws': (0.0650, 0.1750),
            'Rp': 3,
            'Rs': 30,
        }
    },
    'classification': {
        'n_repeats': 10,
        'n_splits': 5,

        'csp_num_components': 4,  # length of feature vector = double of this value
        'csp_norm_trace': False,

    }
}

random.seed(config['random_state'])
np.random.seed(config['random_state'])

config['classification']['classifier'] = SVC(kernel='linear')
