from typing import Union

from sf2utils.sample import Sf2Sample as _Sf2Sample

from dataclasses import dataclass
from typing import Optional


class AttrDict(dict):
    def __init__(self, seq=None, **kwargs):
        if seq is None:
            seq = {}

        super(self.__class__, self).__init__(seq, **kwargs)
        self.__dict__ = self


@dataclass
class WavSample:
    name: Optional[str] = None

    start: Optional[int] = None
    end: Optional[int] = None
    start_loop: Optional[int] = None
    end_loop: Optional[int] = None
    sample_rate: Optional[int] = None
    original_pitch: Optional[int] = None
    pitch_correction: Optional[int] = None

    @property
    def duration(self):
        return self.end - self.start

    @property
    def loop_duration(self):
        return self.end_loop - self.start_loop


ISample = Union[WavSample, _Sf2Sample]
