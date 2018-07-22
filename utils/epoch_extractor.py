import numpy as np
from mne import EpochsArray
from itertools import starmap
from .filter_utils import get_filter_params


def session_mapper(session_data, start_offset, duration):
    raw = session_data['run_0']

    times = raw._data[-1]
    cs = np.where(times > 0)[0]

    this_duration = raw._data.shape[1]
    this_events = np.empty((20, 3), dtype=np.int)
    this_data = np.empty((20, 33, duration))

    for i, x in enumerate(cs):
        y = int(times[x])
        begin_at = x + start_offset
        end_at = begin_at + duration
        trial = raw._data[:, begin_at:end_at]

        this_data[i] = trial
        this_events[i] = [begin_at, 0, y]

    return this_data, this_events, this_duration


def extract_epochs(full_data, config, start=0.0):
    """Extract epochs from full subjects data.

    :param epoch_length: in seconds.
    """

    fps = config['fps']
    duration_of_frames = int(config['epoch_duration'] * fps)
    start_at_frame = int(start * fps)

    assert (len(list(list(full_data.values())[0].keys())) == 1)

    info = list(full_data.values())[0]['run_0'].info.copy()

    inst_data = np.empty((60, 33, duration_of_frames))
    inst_events = np.empty((3, 20, 3), dtype=np.int)
    inst_durations = np.empty((3,), dtype=np.int)  # store the length of sessions

    args = list(map(lambda x: [x, start_at_frame, duration_of_frames], full_data.values()))
    results = starmap(session_mapper, args)

    for i, result in enumerate(results):
        data, events, duration = result
        inst_data[i * 20:(i + 1) * 20, :, :] = data
        inst_events[i] = events
        inst_durations[i] = duration
    # print(type(results))
    # import sys; sys.exit()

    inst_events[1, :, 0] += inst_durations[0]
    inst_events[2, :, 0] += np.sum(inst_durations[0:2])
    inst_events = inst_events.reshape((60, 3))

    this_sub_data = EpochsArray(inst_data, info, events=inst_events, baseline=(None, None))
    this_sub_data.pick_types(eeg=True)

    this_sub_data.set_eeg_reference('average')

    filter_params, filter_dict = get_filter_params(config)
    this_sub_data.filter(*filter_params, **filter_dict)

    return this_sub_data
