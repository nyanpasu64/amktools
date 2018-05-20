#!/usr/bin/env python3
import glob
import logging
import os
import re
import shutil
import sys
import wave
from contextlib import contextmanager
from decimal import Decimal
from fractions import Fraction
from typing import List, Dict

from sf2utils.sample import Sf2Sample
from sf2utils.sf2parse import Sf2File

from amk_tools import wav2brr
from amk_tools.util import AttrDict
from amk_tools.wav2brr import ISample


logging.root.setLevel(logging.ERROR)  # to silence overly pedantic SF2File

VERBOSE = ('--verbose' in sys.argv)
NOWRAP = True


def path_append(*it):
    for el in it:
        os.environ['PATH'] += os.pathsep + el


path_append(os.curdir, r'C:\Program Files (x86)\sox-14-4-2')

# noinspection PyUnresolvedReferences
from plumbum.cmd import sox, brr_encoder, brr_decoder, cmd as _cmd


# TODO command-line paths (import click)
WAV = 'wav/'
WAV2AMK = '../../addmusick-1.1.0-beta/'
PROJECT = 'ds_rr/'


def set_maybe(d, key, value):
    if key not in d:
        d[key] = value


def round_frac(frac):
    try:
        return round(Decimal(frac.numerator) / Decimal(frac.denominator), 20)
    except AttributeError:
        return frac


@contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(previous_dir)


# def cmd(*args, fg=False):
#     """ Invoke via cmd.exe """
#     command = _cmd['/c'][args]
#     if fg:
#         # noinspection PyStatementEffect
#         command & FG
#     else:
#         return command()


total_blocks_regex = re.compile(r'^Size of file to encode : [0-9]+ samples = ([0-9]+) BRR blocks.',
                                re.MULTILINE)
loop_regex = re.compile(
    r'^Position of the loop within the BRR sample : [0-9]+ samples = ([0-9]+) BRR blocks.',
    re.MULTILINE)
iratio_regex = re.compile(r'Resampling by effective ratio of ([0-9.]+)\.\.\.', re.MULTILINE)


def search(regex, s):
    return regex.search(s).group(1)


class Converter:

    def __init__(self, name, wav='.', brr='.'):
        self.name = name
        self.wavname = wav + '/' + name + '.wav'
        self.brrname = brr + '/' + name + '.brr'

        w = wave.open(self.wavname)
        self.rate = w.getframerate()
        self.len = w.getnframes()
        #
        # rate, data = wavfile.read(self.wavname)
        # len_ = len(data)
        # assert rate == self.rate
        # assert len_ == self.len

    def get_len(self):
        # return int(soxi['-s', self.wavname]().strip())
        return self.len

    def get_rate(self):
        # return int(soxi['-r', self.wavname]().strip())
        return self.rate

    # def get_loop_len(self, loop: int):
    #     return self.get_len(self.wavname) - loop

    def attenuate(self, volume: Fraction):
        # TODO: no more sox?
        quiet_name = self.wavname + ' attenuate.wav'

        args = ['-v', str(round_frac(volume)), self.wavname, quiet_name]
        if VERBOSE: print('sox', ' '.join(args))
        sox[args]()

        self.wavname = quiet_name

    def get_ratio(self, loop: int, blocks: Fraction):
        """
        Resample the audio file.
        :param loop: Current loop BEGIN index.
        :param blocks: The final LENGTH of the looped section.
        :return: The final LOOP START INDEX.
        """

        if loop is None:
            loop = 0

        looplen = self.get_len() - loop
        looplen_f = blocks * 16
        ratio = Fraction(looplen_f, looplen)

        return ratio

    def convert(self, ratio: Fraction, loop: int, truncate: int = None, decode: bool = False):
        """
        Convert self.wavname to self.brrname, resampling by ratio.
        :param loop: Loop begin point.
        :param ratio: Resampling ratio.
        :param truncate: End of sample (loop end point).
        :param decode: Whether to decode sample back to wav.
        :return: Effective resampling ratio (TODO -> return type)
        """
        # loop: samples
        # TODO: -a -g
        args = [self.wavname, self.brrname]

        if loop is not None:
            args[0:0] = ['-l' + str(loop)]

        if True:
            args[0:0] = ['-rl' + str(round_frac(1 / ratio))]

        if NOWRAP:
            args[0:0] = ['-w']

        if truncate:
            args[0:0] = ['-t' + str(truncate)]

        output = brr_encoder[args]().replace('\r', '')

        if VERBOSE:
            print('brr_encoder', ' '.join(args))
            print(output)

        # If NOWRAP is True, this should never happen.
        if str(output).find('Caution : Wrapping was used.') != -1:
            if not VERBOSE: print(output)
            assert not NOWRAP
            raise Exception('Wrapping detected!!')

        if loop is not None:
            loop_idx = int(search(loop_regex, output))
            byte_offset = loop_idx * 9
            del loop_idx
        else:
            byte_offset = 0

        ratio = 1 / Fraction(iratio_regex.search(output).group(1))

        if VERBOSE: print('loop_bytes', byte_offset)

        if decode:
            self.decode(ratio)

        with open(self.brrname, 'r+b') as brrfile:
            data = byte_offset.to_bytes(2, 'little') + brrfile.read()

        with open(self.brrname, 'wb') as brrfile:
            brrfile.write(data)

        return ratio

    def decode(self, ratio):
        args = ['-s' + str(round_frac(self.get_rate() * ratio)), self.brrname,
                self.name + ' decoded.wav']
        decode_output = brr_decoder[args]()
        if VERBOSE:
            print(decode_output.replace('\r', ''))


def convert_cfg(cfgname: str, name2sample: 'Dict[str, Sf2Sample]'):
    # Skip tilde-folders.
    if cfgname[0] == '~': return

    name = cfgname[:cfgname.rfind('.')]
    samp_name = name[name.rfind('/') + 1:]

    if VERBOSE: print('~~~~~', name, '~~~~')

    try:
        with open(cfgname) as cfgfile:
            # TODO ruamel.yaml... maybe not, python arithmetic expressions are neat
            config = AttrDict(eval(cfgfile.read()))

        loop = config.get('loop', 'whatever')   # TODO
        truncate = config.get('truncate', None)
        ratio = config.get('ratio', 1)
        volume = config.get('volume', 1)
        transpose = config.get('transpose', 0)
        transpose += config.get('orig-smw', 0)
        at = config.get('at', None)  # MIDI pitch of original note

        # TODO dirty code, refactor into WavSample
        sample = config.get('sample', None)     # type: ISample
        if sample or samp_name not in name2sample:
            sample = AttrDict(sample if sample else {})
            set_maybe(sample, 'pitch_correction', 0)
            set_maybe(sample, 'name', samp_name)
        else:
            sample = name2sample[samp_name]
        if transpose:
            sample.original_pitch -= transpose

        if at is not None:
            sample.original_pitch = at
            sample.pitch_correction = 0
            if loop == 'whatever':
                loop = None

        if loop == 'whatever':
            try:
                loop = sample.start_loop
                truncate = sample.end_loop
            except AttributeError:
                pass

        conv = Converter(name)
        sample.sample_rate = conv.rate

        if volume != 1:
            conv.attenuate(volume)

        # if isinstance(ratio, float):
        #     raise Exception('inaccurate ratio!')
        ratio = Fraction(ratio)
        ratio = conv.convert(ratio=ratio, loop=loop, truncate=truncate, decode=True)
        shutil.copy(conv.brrname, WAV2AMK + 'samples/' + PROJECT)

        wav2brr.brr_tune(sample, ratio)     # calls print

        if VERBOSE: print()

    except Exception:
        print('At file', name, file=sys.stderr)
        raise


def main(name):
    sf2_file = open(name, 'rb')
    sf2 = Sf2File(sf2_file)
    samples = sorted(sf2.samples[:-1], key=lambda s: s.name)  # type: List[Sample]
    name2sample = {sample.name: sample for sample in samples}  # type: Dict[str, Sample]

    # TODO arbitrary directory tree
    with pushd(WAV + WAV2AMK + 'samples/' + PROJECT):
        for wat in glob.glob('*'):
            os.remove(wat)

    with pushd(WAV):
        folders = [f[:-1] for f in glob.glob('*/') if '~' not in f]
        configs = [f[:f.find('\\')] for f in glob.glob('*/*.cfg*') if '~' not in f]

        # Raise exception if empty folders discovered (FIXME what if 2 configs in 1 folder?)
        # wait "configs" is a list of folders containing cfg! this code is hot garbage
        if len(folders) != len(configs):
            raise Exception(set(folders) - set(configs))

        for cfgname in sorted(glob.glob(r'*/*.cfg')):  # type: str
            cfgname = cfgname.replace('\\', '/')
            convert_cfg(cfgname, name2sample)

    # FIXME
    os.system('build.cmd')


if __name__ == '__main__':
    main(sys.argv[1])
