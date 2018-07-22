from scipy import signal
from functools import lru_cache
import mne


@lru_cache(maxsize=None)
def calculate_cheby2_params(Wp, Ws, Rp, Rs, fps):
    _ord, Wn = signal.cheb2ord(Wp, Ws, Rp, Rs)
    b, a = signal.cheby2(_ord, Rs, Wn, btype='bandpass')

    nyquist_f = fps / 2.

    passband = (Wp[0] * nyquist_f, Wp[1] * nyquist_f)
    stopband = (Ws[0] * nyquist_f, Ws[1] * nyquist_f)

    iir_params = mne.filter.construct_iir_filter({
        'b': b,
        'a': a,
        'output': 'ba'}, passband, stopband, fps, btype='bandpass')
    return passband, {'method': 'iir', 'iir_params': iir_params}


def get_filter_params(config):
    fc = config['filter']['ma'] if config['is_ma'] else config['filter']['mi']

    Wp, Ws, Rp, Rs = fc['Wp'], fc['Ws'], fc['Rp'], fc['Rs']
    fps = config['fps']

    return calculate_cheby2_params(Wp, Ws, Rp, Rs, fps)
