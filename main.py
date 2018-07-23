import logging
from config import config

from tasks import by_window, cross_validate_wp, test

mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.ERROR)


def main():
    # by_window(config)
    # cross_validate_wp(config)
    test(config)


if __name__ == '__main__':
    main()
