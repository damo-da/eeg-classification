import logging
import config
import random

from tasks import by_window\

mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.ERROR)


def main():
    random.seed(config.CONFIG['random_state'])
    by_window(config.CONFIG)


if __name__ == '__main__':
    main()
