from sklearn.svm import SVC
import random
import numpy as np

config = {
    'is_ma': True,
    'fps': 200,

    'concurrent': True,
    # 'concurrent': False,

    'random_state': 100,
    # 'random_state': None,

    'subjects': range(1, 3),

    # 'windows': range(-5, 20),
    'windows': range(-4, 2),
    'epoch_duration': 3.0,

    'filter': {
        'ma': {
            'Wp': (0.0400, 0.3500),  # passband
            'Ws': (0.0100, 0.4000),  # stopband
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
        'n_splits': 10,
        'test_size': 0.2,

        'csp_norm_trace': False,

    }
}

random.seed(config['random_state'])
np.random.seed(config['random_state'])

config['classification']['classifier'] = SVC(kernel='linear')
