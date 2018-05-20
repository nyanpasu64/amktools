from fractions import Fraction

from typing import List, Dict
# import resampy
# import numpy as np
# from scipy.io import wavfile

from sf2utils.sf2parse import Sf2File
from sf2utils.instrument import Sf2Instrument
from sf2utils.preset import Sf2Preset
from sf2utils.riffparser import RiffParser, ensure_chunk_size, from_cstr
from sf2utils.sample import Sf2Sample as Sample

# from convert_brr import Converter

FOLDER = 'wav2'


class AttrDict(dict):
    def __init__(self, seq={}, **kwargs):
        super(self.__class__, self).__init__(seq, **kwargs)
        self.__dict__ = self



def freqf(note, cents=0):
    freq = 440 * 2**((note - 69 + cents/100) / 12)
    return freq



def wav_analyze(name='SSEQ_0041.sf2'):
    sf2_file = open(name, 'rb')
    sf2 = Sf2File(sf2_file)
    samples = sorted(sf2.samples[:-1], key=lambda s: s.name)    # type: List[Sample]
    name2sample = {sample.name : sample for sample in samples}  # type: Dict[str, Sample]

    # 'duration'
    # 'start_loop end_loop loop_duration'
    # 'sample_rate original_pitch pitch_correction'

    s = name2sample['bass']
    # s = fakeResample(s, 1 / 2.5)
    brr_tune(s, '0.4', True)


# def fakeResample(sample: Sample, ratio):
#     sample2 = AttrDict({})  # type: Sample
#     for attr in ['name', 'original_pitch', 'pitch_correction']:
#         setattr(sample2, attr, getattr(sample, attr))
#
#     for attr in ['duration', 'start_loop', 'end_loop', 'loop_duration', 'sample_rate']:
#         val = round(getattr(sample, attr) * ratio)
#         setattr(sample2, attr, val)
#
#     sample = sample2
#
#     assert sample.start_loop + sample.loop_duration == sample.end_loop
#     return sample2


def brr_tune(sample: Sample, ratio, loop:bool = None):
    ratio = Fraction(ratio)

    # if sample.name[0] == '_':
    #     raise NotImplementedError
    #     freq = freqf(60)
    # else:
    freq = freqf(sample.original_pitch, sample.pitch_correction)
        # since original_pitch is correct in DS Rainbow Road percussion

    N = (sample.sample_rate * ratio) / freq / 16
    tuning = round(N * 256)
    tuneStr = '$%02x $%02x' % (tuning // 256, tuning % 256)
    print(sample.name, tuneStr)

    # cnv = Converter(sample.name, 'wav', 'brr')
    #
    # if loop:
    #     assert sample.end_loop == sample.duration
    #     cnv.convert(loop=sample.start_loop, ratio=ratio, truncate=sample.end_loop)
    # else:
    #     cnv.convert(ratio=inv_ratio)




#
# def loop_brr(sample: Sample):
#     finalLoopLen = round(sample.loop_duration / 16) * 16
#     ratio = finalLoopLen / sample.loop_duration
#
#     if ratio != 1:
#         assert sample.sample_width == 2
#         wav = np.frombuffer(sample.raw_sample_data, dtype=np.int16)
#
#         # f = 1/λ
#         resampled = resampy.resample(wav, finalLoopLen, sample.loop_duration)
#

# if __name__ == '__main__':
#     wav_analyze()
