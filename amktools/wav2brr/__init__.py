#!/usr/bin/env python3
import logging
import os
import re
import shutil
import sys
import wave
from contextlib import contextmanager
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, IO, Optional
from typing import NamedTuple, Union

import click
from ruamel.yaml import YAML
from sf2utils.sample import Sf2Sample
from sf2utils.sf2parse import Sf2File

from amktools import common
from amktools.wav2brr import tuning
from amktools.wav2brr.tuning import note2ratio
from amktools.wav2brr.util import AttrDict, WavSample, ISample


def path_prepend(*paths: Union[Path, str]):
    prefixes = [str(path) for path in paths]
    prefixes.append(os.environ['PATH'])
    os.environ['PATH'] = os.pathsep.join(prefixes)


if getattr(sys, 'frozen', False):
    app_path = Path(sys._MEIPASS)       # PyInstaller
else:
    app_path = Path(__file__).parent    # python -m

path_prepend(app_path / 'exe')
# os.getcwd() removed, since my bundled version of brr_encoder fixes wrapping
# and we don't want to call old versions ever.


from plumbum import local
brr_encoder = local['brr_encoder']
brr_decoder = local['brr_decoder']

yaml = YAML(typ='safe')
logging.root.setLevel(logging.ERROR)  # to silence overly pedantic SF2File


def rm_recursive(path: Path, optional=False):
    try:
        path.rmdir()
        pass    # branch coverage: this is a directory
    except NotADirectoryError:
        path.unlink()
        pass  # branch coverage: this is a file
    except FileNotFoundError:
        if not optional:
            raise

# Begin command-line parsing

CliOptions = NamedTuple('CliOptions', (
    ('verbose', int),
    ('sample_folder', str),
    ('decode_loops', int)
))


Folder = click.Path(exists=True, file_okay=False)


def ensure_positive(_ctx, param, value):
    if value is None:
        return value
    elif value >= 1:
        return value
    else:
        raise click.BadParameter('%s must be positive' % param)


def pathify(_ctx, _param, value):
    return Path(value)


# We move all files under AMK/samples/ to this subfolder.
BACKUP_ROOT = Path('wav2brr backups')

# Copy these BRR files directly to destination subfolder.
BRR_SOURCE = Path('~brr')


# TODO docs
@click.command()
@click.argument('wav_folder', type=Folder, callback=pathify)
@click.argument('amk_folder', type=Folder, callback=pathify)
@click.argument('sample_subfolder', type=click.Path(), callback=pathify)    # FIXME pick new name
@click.option(
    *'sf2_file --sf2 -S'.split(), type=click.File('rb'))
@click.option(
    *'decode_loops --decode-loops -D'.split(), type=int, callback=ensure_positive)
@click.option(
    *'verbose --verbose -v'.split(), count=True)
def main(wav_folder: Path, amk_folder: Path, sample_subfolder: Path,
         sf2_file: Optional[IO[bytes]], decode_loops: Optional[int], verbose: int):

    # Compute sample output folder

    sample_root = (amk_folder / 'samples').resolve()
    sample_folder = (sample_root / sample_subfolder).resolve()

    if sample_root not in sample_folder.parents:
        raise click.BadParameter('Sample folder "{}" is not within "{}"\n\n'.format(sample_subfolder, sample_root) +
                                 'If you use samples/ without a subfolder, all files will be cleared!\n')

    if not sample_folder.exists():
        print('Creating sample folder', sample_folder)
        os.mkdir(str(sample_folder))

    # Begin wav2brr setup

    if decode_loops is None:
        decode_loops = 1

    opt = CliOptions(
        verbose=verbose,
        sample_folder=str(sample_folder),
        decode_loops=decode_loops)

    if sf2_file is not None:
        sf2 = Sf2File(sf2_file)
        samples = sorted(sf2.samples[:-1], key=lambda s: s.name)  # type: List[Sf2Sample]
        name2sample = {sample.name: sample for sample in samples}  # type: Dict[str, Sf2Sample]
    else:
        name2sample = {}

    # Clear old samples

    with pushd(sample_folder):
        # BRR_BACKUP
        backup_root = BACKUP_ROOT
        backup_root.mkdir(exist_ok=True)

        # file.brr or BRR_BACKUP
        for old in Path().glob('*'):
            if old.is_file():   # old != backup_root:
                # remove BRR_BACKUP/file.brr
                backup_dest = backup_root / old

                rm_recursive(backup_dest, optional=True)
                old.rename(backup_dest)

    # Create new samples

    tunings = {}
    with pushd(wav_folder):
        p = Path()
        folders = sorted(x for x in p.iterdir()
                         if x.is_dir() and '~' not in x.name)

        for folder in folders:
            cfgs = sorted(x for x in folder.glob('*.cfg') if '~' not in x.name)
            if len(cfgs) != 1:
                fnames = [cfg.name for cfg in cfgs]
                raise ValueError(
                    'folder "{}" must contain one .cfg, currently contains {}'
                        .format(folder, fnames)
                )

            cfg_path = str(cfgs[0]).replace('\\', '/')
            name, tune = convert_cfg(opt, cfg_path, name2sample)
            tunings[name] = tune

        # Copy over .brr files directly
        for brr in BRR_SOURCE.glob('*.brr'):
            shutil.copy(str(brr), opt.sample_folder)


    with open(common.TUNING_PATH, 'w') as f:
        yaml.dump(tunings, f)


# **** .cfg file parsing ****

def convert_cfg(opt: CliOptions, cfg_path: str, name2sample: 'Dict[str, Sf2Sample]'):
    """
    :return: (Path to BRR file, tuning string)
    """
    cfg_prefix = os.path.splitext(cfg_path)[0]      # folder/cfg
    cfg_fname = os.path.basename(cfg_prefix)        # cfg

    if opt.verbose: print('~~~~~', cfg_prefix, '~~~~')

    try:
        with open(cfg_path) as cfgfile:
            cfg = AttrDict(eval(cfgfile.read()))

        rate = cfg.get('rate', None)    # sampling rate
        ratio = cfg.get('ratio', 1)
        volume = cfg.get('volume', 1)
        transpose = cfg.get('transpose', 0)
        at = cfg.get('at', None)  # MIDI pitch of original note
        # Load resampling settings.

        if cfg_fname in name2sample:
            sample = name2sample[cfg_fname]     # type: ISample
        else:
            sample = WavSample()
            sample.pitch_correction = 0
            sample.name = cfg_fname
            # All other attributes/fields default to None

        # Transpose sample.

        if transpose:
            sample.original_pitch -= transpose

        if at is not None:
            sample.original_pitch = at
            sample.pitch_correction = 0

        # Loop sample.
        # this is fucking fizzbuzz
        if {'loop', 'truncate'} & cfg.keys():
            loop = truncate = None      # type: Optional[int]
            if 'loop' in cfg:
                loop = cfg['loop']
            if 'truncate' in cfg:
                truncate = cfg['truncate']
        else:
            loop = sample.start_loop
            truncate = sample.end_loop


        # Convert sample.

        conv = Converter(opt, cfg_prefix, transpose=transpose)
        sample.sample_rate = rate or conv.rate

        ratio = Fraction(ratio)
        ratio = conv.convert(ratio=ratio, loop=loop, truncate=truncate, volume=volume, decode=True)
        shutil.copy(conv.brrname, opt.sample_folder)

        tune = tuning.brr_tune(sample, ratio)[1]
        print(cfg_fname, tune)

        if opt.verbose: print()

        return cfg_fname + BRR_EXT, tune

    except Exception:
        print('At file', cfg_prefix, file=sys.stderr)
        raise


# **** WAV to BRR conversion ****


def decimal_repr(num):
    # ugly, unnecessary, but it works.

    if isinstance(num, Fraction):
        num = round(Decimal(num.numerator) / Decimal(num.denominator), 20)

    return str(num)


@contextmanager
def pushd(new_dir: Union[Path, str]):
    previous_dir = os.getcwd()
    os.chdir(str(new_dir))
    yield
    os.chdir(previous_dir)


loop_regex = re.compile(
    r'^Position of the loop within the BRR sample : \d+ samples = (\d+) BRR blocks\.',
    re.MULTILINE)
reciprocal_ratio_regex = re.compile(
    r'Resampling by effective ratio of ([\d.]+)\.\.\.', re.MULTILINE)
# Do not remove the trailing ellipses. That will hide bugs where the resampling
# ratio is not extracted correctly (eg. truncated at the decimal point).


def search(regex, s):
    return regex.search(s).group(1)


WAV_EXT = '.wav'
BRR_EXT = '.brr'

NOWRAP = True

# TODO: "name" actually means "file path minus extension".
class Converter:
    def __init__(self, opt: CliOptions, name, transpose=0):
        """
        :param opt: Command-line options (including .brr output paths),
            shared across samples.
        :param name: Path to .wav file, without file extension.
        :param transpose: Semitones to transpose (can be float)
        """
        self.opt = opt

        self.name = name
        self.wavname = name + WAV_EXT
        self.brrname = name + BRR_EXT
        self.transpose = transpose

        w = wave.open(self.wavname)
        self.rate = w.getframerate()
        self.len = w.getnframes()


    def get_len(self):
        return self.len

    def get_rate(self):
        return self.rate

    def convert(self, ratio: Fraction, loop: Optional[int], truncate: Optional[int],
                volume: Fraction, decode: bool) -> Fraction:
        """
        Convert self.wavname to self.brrname, resampling by ratio.
        :param ratio: Resampling ratio.
        :param loop: Loop begin point.
        :param truncate: End of sample (loop end point).
        :param volume: Volume to multiply sample by.
        :param decode: Whether to decode sample back to wav.
        :return: Effective resampling ratio
        """
        opt = self.opt

        args = ['-g', self.wavname, self.brrname]

        if NOWRAP:  # always true
            args[0:0] = ['-w']

        # Loop and truncate
        is_loop = (loop is not None)
        if is_loop:
            args[0:0] = ['-l' + str(loop)]
        if truncate is not None:
            args[0:0] = ['-t' + str(truncate)]

        # Resample
        """
        Even if ratio=1, encoder may resample slightly, to ensure loop is
        multiple of 16. So enable bandlimited sinc to preserve high frequencies.
        NOTE: Default linear interpolation is simple, but is garbage at
        preserving high frequencies.
        """
        args[0:0] = ['-rb' + decimal_repr(1 / ratio)]

        # Attenuate volume
        if volume != 1:
            args[:0] = ['-a' + decimal_repr(volume)]

        # **** Call brr_encoder ****
        output = brr_encoder[args]().replace('\r', '')

        if opt.verbose:
            print('brr_encoder', ' '.join(args))
            print(output)

        if str(output).find('Caution : Wrapping was used.') != -1:
            if not opt.verbose: print(output)
            # If NOWRAP is True, this should never happen.
            assert not NOWRAP
            raise Exception('Wrapping detected!!')

        if is_loop:
            loop_idx = int(search(loop_regex, output))
        else:
            loop_idx = 0
        byte_offset = loop_idx * 9

        wav2brr_ratio = 1 / Fraction(reciprocal_ratio_regex.search(output).group(1))

        if opt.verbose: print('loop_bytes', byte_offset)

        if decode:
            self.decode(wav2brr_ratio, loop_idx if is_loop else None)

        with open(self.brrname, 'r+b') as brr_file:
            data = byte_offset.to_bytes(2, 'little') + brr_file.read()

        with open(self.brrname, 'wb') as brr_file:
            brr_file.write(data)

        return wav2brr_ratio

    def decode(self, ratio, loop_idx):
        opt = self.opt

        rate = self.get_rate() * ratio * note2ratio(self.transpose)
        args = ['-g', '-s' + decimal_repr(rate), self.brrname,
                self.name + ' decoded.wav']
        if loop_idx is not None:
            args[:0] = ['-l{}'.format(loop_idx), '-n{}'.format(opt.decode_loops)]
        decode_output = brr_decoder[args]()
        if opt.verbose:
            # print(brr_decoder[args]) is hard to read since it uses full EXE path
            print('brr_decoder', ' '.join(args))
            print(decode_output.replace('\r', ''))
