import sklearn

CONFIG = {
    'is_ma': True,
    'fps': 200,

    'subjects': range(1, 5),

    'windows': range(-5, 20),
    'epoch_duration': 3.0,

    'filter': {
        'ma': {
            'Wp': (0.0400, 0.3500),
            'Ws': (0.0100, 0.3800),
            'Rp': 3,
            'Rs': 30,
            'passband': (4, 35),
            'stopband': (0, 40)
        },
        'mi': {
            'Wp': (0.0950, 0.1450),
            'Ws': (0.0650, 0.1750),
            'Rp': 3,
            'Rs': 30,
            'passband': (8, 30),
            'stopband': (0, 40)
        }
    },
    'classification': {
        'n_splits': 10,
        'test_size': 0.2,

        'csp_norm_trace': False,
        'classifier': sklearn.svm.SVC(kernel='linear')

    }
}
