import logging
import config

from tasks import by_window, cross_validate_wp, apply_algorithm

mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.ERROR)


def main():
    apply_algorithm(by_window, config.CONFIG)
    # apply_algorithm(cross_validate_wp, config.CONFIG)


if __name__ == '__main__':
    main()
