from fractions import Fraction

from amktools.wav2brr.util import ISample as _ISample


def note2ratio(note, cents=0):
    """ Converts semitones to a frequency ratio. """
    ratio = 2 ** ((note + cents / 100) / 12)
    return ratio


def note2pitch(note, cents=0):
    """ Converts a MIDI note to an absolute frequency (Hz). """
    freq = 440 * note2ratio(note - 69, cents)
    return freq


def brr_tune(sample: _ISample, ratio):
    ratio = Fraction(ratio)
    freq = note2pitch(sample.original_pitch, sample.pitch_correction)
    # Absolute frequencies unsupported. (original_pitch is correct in DS Rainbow Road percussion)

    N = (sample.sample_rate * ratio) / freq / 16
    tuning = round(N * 256)
    tuneStr = '$%02x $%02x' % (tuning // 256, tuning % 256)
    return sample.name, tuneStr
