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

from amktools.wav2brr import tuning
from amktools.wav2brr.util import AttrDict
from amktools.wav2brr.tuning import note2ratio


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
PROJECT = 'idol1 final hours/'


def set_maybe(d, key, value):
    # TODO remove this function
    if key not in d:
        d[key] = value


def round_frac(frac):
    # TODO remove this function
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
reciprocal_ratio_regex = re.compile(r'Resampling by effective ratio of ([0-9.]+)\.\.\.', re.MULTILINE)


def search(regex, s):
    return regex.search(s).group(1)


class Converter:

    def __init__(self, name, wav='.', brr='.', transpose=0):
        self.name = name
        self.wavname = wav + '/' + name + '.wav'
        self.brrname = brr + '/' + name + '.brr'
        self.transpose = transpose

        w = wave.open(self.wavname)
        self.rate = w.getframerate()
        self.len = w.getnframes()


    def get_len(self):
        return self.len

    def get_rate(self):
        return self.rate

    # def get_loop_len(self, loop: int):
    #     return self.get_len(self.wavname) - loop

    def attenuate(self, volume: Fraction):
        # TODO: eliminate dependency on sox, using -a flag
        quiet_name = self.wavname + ' attenuate.wav'

        args = ['-v', str(round_frac(volume)), self.wavname, quiet_name]
        if VERBOSE: print('sox', ' '.join(args))
        sox[args]()

        self.wavname = quiet_name

    # def get_ratio(self, loop: int, blocks: Fraction):
    #     """
    #     Resample the audio file.
    #     :param loop: Current loop BEGIN index.
    #     :param blocks: The final LENGTH of the looped section.
    #     :return: The final LOOP START INDEX.
    #     """
    #
    #     if loop is None:
    #         loop = 0
    #
    #     looplen = self.get_len() - loop
    #     looplen_f = blocks * 16
    #     ratio = Fraction(looplen_f, looplen)
    #
    #     return ratio

    def convert(self, ratio: Fraction, loop: int, truncate: int = None, decode: bool = False) -> Fraction:
        """
        Convert self.wavname to self.brrname, resampling by ratio.
        :param ratio: Resampling ratio.
        :param loop: Loop begin point.
        :param truncate: End of sample (loop end point).
        :param decode: Whether to decode sample back to wav.
        :return: Effective resampling ratio
        """
        # TODO: -a attenuation, -g gaussian?
        args = ['-g', self.wavname, self.brrname]

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

        wav2brr_ratio = 1 / Fraction(reciprocal_ratio_regex.search(output).group(1))

        if VERBOSE: print('loop_bytes', byte_offset)

        if decode:
            self.decode(wav2brr_ratio)

        with open(self.brrname, 'r+b') as brr_file:
            data = byte_offset.to_bytes(2, 'little') + brr_file.read()

        with open(self.brrname, 'wb') as brr_file:
            brr_file.write(data)

        return wav2brr_ratio

    def decode(self, ratio):
        rate = self.get_rate() * ratio * note2ratio(self.transpose)
        args = ['-s' + str(round_frac(rate)), self.brrname,
                self.name + ' decoded.wav']
        decode_output = brr_decoder[args]()
        if VERBOSE:
            print(decode_output.replace('\r', ''))


def convert_cfg(cfg_path: str, name2sample: 'Dict[str, Sf2Sample]'):
    # Skip tilde-folders.
    if cfg_path[0] == '~': return

    name = cfg_path[:cfg_path.rfind('.')]   # folder/cfg
    cfg_name = name[name.rfind('/') + 1:]   # cfg

    if VERBOSE: print('~~~~~', name, '~~~~')

    try:
        with open(cfg_path) as cfgfile:
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
        # FIXME sample = config.get('sample', None)     # type: ISample
        sample = None
        if bool(sample) or cfg_name not in name2sample:     # wtf
            sample = AttrDict(sample or {})                 # wtf
            set_maybe(sample, 'pitch_correction', 0)        # burn set_maybe with fire
            set_maybe(sample, 'name', cfg_name)
        else:
            sample = name2sample[cfg_name]
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

        conv = Converter(name, transpose=transpose)
        sample.sample_rate = conv.rate

        if volume != 1:
            conv.attenuate(volume)

        # if isinstance(ratio, float):
        #     raise Exception('inaccurate ratio!')
        ratio = Fraction(ratio)
        ratio = conv.convert(ratio=ratio, loop=loop, truncate=truncate, decode=True)    # FIXME command line args
        shutil.copy(conv.brrname, WAV2AMK + 'samples/' + PROJECT)

        tuning.brr_tune(sample, ratio)     # calls print

        if VERBOSE: print()

    except Exception:
        print('At file', name, file=sys.stderr)
        raise


def main(sf2_name):
    if sf2_name is not None:
        sf2_file = open(sf2_name, 'rb')
        sf2 = Sf2File(sf2_file)
        samples = sorted(sf2.samples[:-1], key=lambda s: s.name)  # type: List[Sf2Sample]
        name2sample = {sample.name: sample for sample in samples}  # type: Dict[str, Sf2Sample]
    else:
        name2sample = {}

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

        for cfg_path in sorted(glob.glob(r'*/*.cfg')):  # type: str
            cfg_path = cfg_path.replace('\\', '/')
            convert_cfg(cfg_path, name2sample)

    # FIXME
    # os.system('build.cmd')


if __name__ == '__main__':
    if 1 < len(sys.argv):
        main(sys.argv[1])
    else:
        main(None)
