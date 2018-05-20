from fractions import Fraction
from typing import Union

from sf2utils.sample import Sf2Sample

from amk_tools.util import WavSample


ISample = Union[WavSample, Sf2Sample]


def freqf(note, cents=0):
    freq = 440 * 2 ** ((note - 69 + cents / 100) / 12)
    return freq


def brr_tune(sample: ISample, ratio):
    ratio = Fraction(ratio)
    freq = freqf(sample.original_pitch, sample.pitch_correction)
    # since original_pitch is correct in DS Rainbow Road percussion

    N = (sample.sample_rate * ratio) / freq / 16
    tuning = round(N * 256)
    tuneStr = '$%02x $%02x' % (tuning // 256, tuning % 256)
    print(sample.name, tuneStr)
