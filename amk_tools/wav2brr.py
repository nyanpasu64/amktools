from fractions import Fraction
from typing import Union

from sf2utils.sample import Sf2Sample

from amk_tools.util import WavSample


# TODO confusing filename, tuning.py?

ISample = Union[WavSample, Sf2Sample]


def note2ratio(note, cents=0):
    ratio = 2 ** ((note + cents / 100) / 12)
    return ratio


def note2pitch(note, cents=0):
    freq = 440 * note2ratio(note - 69, cents)
    return freq


def brr_tune(sample: ISample, ratio):
    ratio = Fraction(ratio)
    freq = note2pitch(sample.original_pitch, sample.pitch_correction)
    # FIXME since original_pitch is correct in DS Rainbow Road percussion

    N = (sample.sample_rate * ratio) / freq / 16
    tuning = round(N * 256)
    tuneStr = '$%02x $%02x' % (tuning // 256, tuning % 256)
    print(sample.name, tuneStr)
