import itertools
from utils import single_param_cross_validator_func
from utils import apply_algorithm, plot_cv_scores

_cv_wp_range1 = (0.02, 0.03)
_cv_wp_range2 = (0.35, 0.40)
_cv_wp_values = list(itertools.product(_cv_wp_range1, _cv_wp_range2))

_cv_wp_label = 'Wp'


def config_maper(cfg, val):
    cfg['filter']['ma']['Wp'] = val
    cfg['filter']['ma']['Ws'] = (val[0] - 0.01, val[1] + 0.05)


def cross_validate_wp_func(subject, config, dataset):
    return single_param_cross_validator_func(subject, config, dataset, _cv_wp_label,
                                             values=_cv_wp_values,
                                             apply_func=config_maper)


def cross_validate_wp(config):
    result = apply_algorithm(cross_validate_wp_func, config)
    plot_cv_scores(result, _cv_wp_label, _cv_wp_values, config)
