from typing import Union

from sf2utils.sample import Sf2Sample as _Sf2Sample


class AttrDict(dict):
    def __init__(self, seq=None, **kwargs):
        if seq is None:
            seq = {}

        super(self.__class__, self).__init__(seq, **kwargs)
        self.__dict__ = self


class WavSample:
    def __init__(self):
        self.name = None

        self.start = None
        self.end = None
        self.start_loop = None
        self.end_loop = None
        self.sample_rate = None
        self.original_pitch = None
        self.pitch_correction = None

    @property
    def duration(self):
        return self.end - self.start

    @property
    def loop_duration(self):
        return self.end_loop - self.start_loop


ISample = Union[WavSample, _Sf2Sample]
