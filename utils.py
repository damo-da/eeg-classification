import numpy as np
from mne import EpochsArray


def extract_epochs(full_data, subjects, epoch_length=20, fps=200):
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

        this_sub_data = EpochsArray(inst_data, info, events=inst_events)
        all_data[subject] = this_sub_data

    return all_data