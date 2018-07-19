import numpy as np
from mne import EpochsArray
import mne
from scipy import signal
from functools import lru_cache


@lru_cache(maxsize=None)
def calculate_cheby2_params(Wp, Ws, Rp, Rs, passband, stopband, fps):
    _ord, Wn = signal.cheb2ord(Wp, Ws, Rp, Rs)
    b, a = signal.cheby2(_ord, Rs, Wn, btype='bandpass')

    iir_params = mne.filter.construct_iir_filter({
        'b': b,
        'a': a,
        'output': 'ba'}, passband, stopband, fps, btype='bandpass')
    return passband, {'method': 'iir', 'iir_params': iir_params}


def get_filter_params(config):
    fc = config['filter']['ma'] if config['is_ma'] else config['filter']['mi']

    passband = fc['passband']
    stopband = fc['stopband']
    Wp, Ws, Rp, Rs = fc['Wp'], fc['Ws'], fc['Rp'], fc['Rs']
    fps = config['fps']

    return calculate_cheby2_params(Wp, Ws, Rp, Rs, passband, stopband, fps)


def extract_epochs(full_data, subject, config, start=0.0):
    """Extract epochs from full subjects data.

    :param epoch_length: in seconds.
    """

    fps = config['fps']
    duration = config['epoch_duration']

    all_data = {}
    duration_of_frames = int(duration * fps)
    start_at_frame = int(start * fps)

    info = None
    this_sub_data = {'items': []}
    inst_data = np.zeros((60, 33, duration_of_frames))
    inst_events = np.zeros((60, 3), dtype=np.int)

    session_ctr = 0
    time_sum = 0

    for session_id, session_data in full_data.items():
        raw = session_data['run_0']

        info = raw.info

        times = raw._data[-1]
        cs = np.where(times > 0)[0]

        for i, x in enumerate(cs):
            y = int(times[x])
            begin_at = x + start_at_frame
            end_at = begin_at + duration_of_frames
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
    this_sub_data.pick_types(eeg=True)

    this_sub_data.set_eeg_reference('average')

    filter_params, filter_dict = get_filter_params(config)
    this_sub_data.filter(*filter_params, **filter_dict)

    return this_sub_data
