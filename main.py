import logging
import config

from tasks import by_window\

mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.ERROR)


def main():
    # by_window(config.CONFIG)

    cross_validate_wp_ws_rp_rs(config.CONFIG)


if __name__ == '__main__':
    main()
