import numpy as np
from mne import EpochsArray
import mne
from scipy import signal
from functools import lru_cache


def calculate_cheby2_params(using_MA=True, data_type='EEG'):
    if using_MA:
        Wp = (0.0400, 0.3500)
        Ws = (0.0100, 0.3800)
        Rp = 3
        Rs = 30
    else:
        Wp = (0.0950, 0.1450)
        Ws = (0.0650, 0.1750)
        Rp = 3
        Rs = 30

    ord, Wn = signal.cheb2ord(Wp, Ws, Rp, Rs)
    b, a = signal.cheby2(ord, Rs, Wn, btype='bandpass')
    return b, a


@lru_cache(maxsize=None)
def get_filter_params(fps, using_ma):

    pass_band = (4, 35) if using_ma else (8, 30)
    stop_band = (0, 40)

    b, a = calculate_cheby2_params(using_MA=using_ma)

    iir_params = mne.filter.construct_iir_filter({
            'a': a,
            'b': b,
            'output': 'ba'}, pass_band, stop_band, fps, btype='bandpass')
    # iir_params = mne.filter.create_filter(raw._data, sfreq=200, l_freq=4, method='iir', h_freq=35,
    # iir_params=iir_params)
    return pass_band, {'method':'iir', 'iir_params':iir_params}


def extract_epochs(full_data, subjects, epoch_length=20, fps=200, using_ma=True):
    """Extract epochs from full subjects data.

    :param epoch_length: in seconds.
    """

    all_data = {}
    duration = int(epoch_length * fps)
    for subject in subjects:
        info = None
        this_sub_data = {'items': []}
        inst_data = np.zeros((60, 33, duration))
        inst_events = np.zeros((60, 3), dtype=np.int)


        session_ctr = 0
        time_sum = 0
        for session_id, session_data in full_data[subject].items():
            raw = session_data['run_0']

            filter_params, filter_dict = get_filter_params(fps, using_ma)
            raw.filter(*filter_params, **filter_dict)

            info = raw.info

            times = raw._data[-1]
            cs = np.where(times > 0)[0]

            for i, x in enumerate(cs):
                y = int(times[x])
                begin_at = x
                end_at = begin_at + duration
                trial = raw._data[:, begin_at:end_at]

                inst_index = session_ctr * 20 + i

                inst_data[inst_index] = trial
                inst_events[inst_index] = [time_sum + begin_at, 0, y]

                this_sub_data['items'].append({
                    'y': y,
                    'x': trial,
                    'begin_at': begin_at
                })
            session_ctr += 1
            time_sum += raw._data.shape[1]

        this_sub_data = EpochsArray(inst_data, info, events=inst_events, baseline=(None, None))
        all_data[subject] = this_sub_data

    return all_data