import itertools
from .utils import single_param_cross_validator


def cross_validate_rs(config):
    def this_func(cfg, val):
        cfg['filter']['ma']['Rs'] = val

    return single_param_cross_validator(config, 'rs',
                                        ranges=(29, 30, 31, 32, 33, 34, 35, 40, 50),
                                        apply_func=this_func)


def cross_validate_rp(config):
    def this_func(cfg, val):
        cfg['filter']['ma']['Rp'] = val

    return single_param_cross_validator(config, 'rp',
                                        ranges=(1, 2, 3),
                                        apply_func=this_func)


def cross_validate_wp(config):
    def this_func(cfg, val):
        cfg['filter']['ma']['Wp'] = val
        cfg['filter']['ma']['Ws'] = (val[0] - 0.01, val[1] + 0.05)

    range1=(0.02, 0.03, 0.04, 0.05, 0.07)
    range2=(0.35, 0.40)

    return single_param_cross_validator(config, 'wp',
                                        ranges=list(itertools.product(range1, range2)),
                                        apply_func=this_func)


