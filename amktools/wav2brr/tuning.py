from fractions import Fraction
from typing import Optional

from amktools.wav2brr.util import ISample as _ISample


def note2ratio(note, cents=0):
    """ Converts semitones to a frequency ratio. """
    ratio = 2 ** ((note + cents / 100) / 12)
    return ratio


def note2pitch(note, cents=0):
    """ Converts a MIDI note to an absolute frequency (Hz). """
    freq = 440 * note2ratio(note - 69, cents)
    return freq


def brr_tune(sample: _ISample, ratio, tuning: Optional[float]):
    ratio = Fraction(ratio)

    if tuning is not None:
        tuning = tuning * ratio
    else:
        # If a sample is played back N cents flats, the sample is N cents sharp.
        pitch_error = -sample.pitch_correction

        # Note frequency
        cyc_s = note2pitch(sample.original_pitch, pitch_error)
        # Absolute frequencies unsupported. (original_pitch is correct in DS Rainbow Road percussion)

        # Sampling rate
        smp_s = sample.sample_rate * ratio

        # Period nsamp
        smp_cyc = smp_s / cyc_s

        tuning = smp_cyc / 16

    # 8.8 fixed-point value = 256*tuning + fraction
    tuning_int = round(tuning * 256)

    tune_str = '$%02x $%02x' % (tuning_int // 256, tuning_int % 256)
    return sample.name, tune_str
