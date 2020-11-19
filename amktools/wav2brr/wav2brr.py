#!/usr/bin/env python3
import ctypes
import logging
import os
import re
import sys
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, IO, Optional
from typing import Union

import click
from plumbum import local
from ruamel.yaml import YAML
from sf2utils.sample import Sf2Sample
from sf2utils.sf2parse import Sf2File

from amktools.wav2brr import tuning
from amktools.wav2brr.tuning import note2ratio
from amktools.wav2brr.util import AttrDict, WavSample, ISample


def path_prepend(*paths: Union[Path, str]):
    prefixes = [str(path) for path in paths]
    prefixes.append(os.environ["PATH"])
    os.environ["PATH"] = os.pathsep.join(prefixes)


if getattr(sys, "frozen", False):
    app_path = Path(sys._MEIPASS)  # PyInstaller
else:
    app_path = Path(__file__).parent  # python -m

prefix = app_path / "exe"
# os.getcwd() removed, since my bundled version of brr_encoder fixes wrapping
# and we don't want to call old versions ever.

e = str(prefix / "brr_encoder.exe")
d = str(prefix / "brr_decoder.exe")

if os.name != "nt":
    local = local["wine"]

brr_encoder = local[e]
brr_decoder = local[d]

yaml = YAML(typ="safe")
logging.root.setLevel(logging.ERROR)  # to silence overly pedantic SF2File


def rm_recursive(path: Path, optional=False):
    try:
        path.rmdir()
        pass  # branch coverage: this is a directory
    except NotADirectoryError:
        path.unlink()
        pass  # branch coverage: this is a file
    except FileNotFoundError:
        if not optional:
            raise


# Begin command-line parsing


@dataclass
class CliOptions:
    verbose: int
    decode_loops: int


def decimal_repr(num):
    # ugly, unnecessary, but it works.

    if isinstance(num, Fraction):
        num = round(Decimal(num.numerator) / Decimal(num.denominator), 20)

    return str(num)


def coalesce(*args):
    if len(args) == 0:
        raise TypeError("coalesce expected >=1 argument, got 0")
    for arg in args:
        if arg is not None:
            return arg
    raise TypeError("coalesce() called with all None")


@contextmanager
def pushd(new_dir: Union[Path, str]):
    previous_dir = os.getcwd()
    os.chdir(str(new_dir))
    try:
        yield
    finally:
        os.chdir(previous_dir)


end_minus_one_regex = re.compile(
    r"^Size of file to encode : \d+ samples = (\d+) BRR blocks\.", re.MULTILINE
)
loop_regex = re.compile(
    r"^Position of the loop within the BRR sample : \d+ samples = (\d+) BRR blocks\.",
    re.MULTILINE,
)
reciprocal_ratio_regex = re.compile(
    r"Resampling by effective ratio of ([\d.]+)\.\.\.", re.MULTILINE
)
# Do not remove the trailing ellipses. That will hide bugs where the resampling
# ratio is not extracted correctly (eg. truncated at the decimal point).


def search(regex, s):
    return regex.search(s).group(1)


WAV_EXT = ".wav"
BRR_EXT = ".brr"
WAV2BRR_DIRNAME = "~wav2brr"

NOWRAP = False


@dataclass
class BrrResult:
    ratio: Fraction
    loop_samp: int
    nsamp: int


class ConvertSession:
    def __init__(self, opt: CliOptions, wav_path: Path, brr_path: Path, transpose=0):
        """
        :param opt: Command-line options (including .brr output paths),
            shared across samples.
        :param transpose: Semitones to transpose (can be float)
        TODO fill other parameters
        """

        self.opt = opt

        self.wav_path = wav_path
        self.brr_path = brr_path
        self.brr_dir = brr_path.parent

        # .brr and decoded .wav are written to an intermediate folder.
        self.brr_stem = brr_path.stem

        # self.brr_path = str(self.wav2brr_dir / (self.stem + BRR_EXT))
        self.transpose = transpose

        w = wave.open(str(self.wav_path))
        self.rate = w.getframerate()
        self.len = w.getnframes()

    def get_len(self):
        return self.len

    def get_rate(self):
        return self.rate

    def convert(
        self,
        ratio: Fraction,
        loop: Optional[int],
        truncate: Optional[int],
        volume: Fraction,
        decode: bool,
    ) -> BrrResult:
        """
        Convert self.wav_path to self.brr_path, resampling by ratio.
        :param ratio: Resampling ratio.
        :param loop: Loop begin point.
        :param truncate: End of sample (loop end point).
        :param volume: Volume to multiply sample by.
        :param decode: Whether to decode sample back to wav.
        :return: Effective resampling ratio
        """
        opt = self.opt

        args = []

        if True:
            args += ["-g"]

        args += [self.wav_path, self.brr_path]

        if NOWRAP:
            args[0:0] = ["-w"]

        # Loop and truncate
        is_loop = loop is not None
        if is_loop:
            args[0:0] = ["-l" + str(loop)]
        if truncate is not None:
            args[0:0] = ["-t" + str(truncate)]

        # Resample
        """
        Even if ratio=1, encoder may resample slightly, to ensure loop is
        multiple of 16. So enable bandlimited sinc to preserve high frequencies.
        NOTE: Default linear interpolation is simple, but is garbage at
        preserving high frequencies.
        """
        args[0:0] = ["-rb" + decimal_repr(1 / ratio)]

        # Attenuate volume
        if volume != 1:
            args[:0] = ["-a" + decimal_repr(volume)]

        # **** Call brr_encoder ****
        output = brr_encoder[args]().replace("\r", "")

        if opt.verbose:
            print("brr_encoder", " ".join(args))
            print(output)

        # Parse stdout
        if is_loop:
            loop_idx = int(search(loop_regex, output))
        else:
            loop_idx = 0
        byte_offset = loop_idx * 9

        wav2brr_ratio = 1 / Fraction(search(reciprocal_ratio_regex, output))

        end_idx = int(search(end_minus_one_regex, output)) + 1

        # Done
        if opt.verbose:
            print("loop_bytes", byte_offset)

        if decode:
            self.decode(wav2brr_ratio, loop_idx if is_loop else None)

        with open(self.brr_path, "r+b") as brr_file:
            data = byte_offset.to_bytes(2, "little") + brr_file.read()

        with open(self.brr_path, "wb") as brr_file:
            brr_file.write(data)

        # Result

        return BrrResult(wav2brr_ratio, loop_idx * 16, end_idx * 16)

    def decode(self, ratio, loop_idx):
        opt = self.opt

        rate = self.get_rate() * ratio * note2ratio(self.transpose)
        decoded_path = str(self.brr_dir / (self.brr_stem + " decoded.wav"))
        args = ["-s" + decimal_repr(rate), self.brr_path, decoded_path]
        if loop_idx is not None:
            args[:0] = ["-l{}".format(loop_idx), "-n{}".format(opt.decode_loops)]
        decode_output = brr_decoder[args]()
        if opt.verbose:
            # print(brr_decoder[args]) is hard to read since it uses full EXE path
            print("brr_decoder", " ".join(args))
            print(decode_output.replace("\r", ""))


@dataclass
class ConvertResult:
    tuning: str


def convert_cfg(
    opt: CliOptions,
    wav_path: Path,
    brr_path: Path,
    cfg_reader: IO[str],
    name_to_sample: "Dict[str, Sf2Sample]",
) -> ConvertResult:
    cfg = AttrDict(eval(cfg_reader.read()))

    wav_stem = wav_path.stem

    # Input WAV rate
    rate = cfg.get("rate", None)  # Input WAV sampling rate
    detune = cfg.get("detune", None)

    # Resampling
    ratio = cfg.get("ratio", 1)
    target_rate = cfg.get("target_rate", None)

    volume = cfg.get("volume", 1)
    transpose = cfg.get("transpose", 0)
    at = cfg.get("at", None)  # MIDI pitch of original note

    # Exact AMK tuning (wavelength in 16-sample units)
    tuning_ = cfg.get("tuning", None)

    ncyc = cfg.get("ncyc", None)

    # Load resampling settings.

    # https://github.com/pallets/click/issues/188
    if wav_stem in name_to_sample:
        sample = name_to_sample[wav_stem]  # type: ISample
        # sf2utils bug #2 treats negative pitch_correction as signed
        sample.pitch_correction = ctypes.c_int8(sample.pitch_correction).value

    else:
        sample = WavSample()
        sample.pitch_correction = 0
        sample.name = wav_stem
    # All other attributes/fields default to None

    # Transpose sample.

    if transpose:
        sample.original_pitch -= transpose

    if detune is not None:
        sample.pitch_correction = detune

    if at is not None:
        sample.original_pitch = at

    # Loop sample.
    # this is fucking fizzbuzz
    if {"loop", "truncate"} & cfg.keys():
        loop = truncate = None  # type: Optional[int]
        if "loop" in cfg:
            loop = cfg["loop"]
        if "truncate" in cfg:
            truncate = cfg["truncate"]
    elif sample.start_loop == sample.end_loop:  # both 0 in vgmtrans sf2
        loop = None
        truncate = None
    else:
        loop = sample.start_loop
        truncate = sample.end_loop

    print(f"MIDI at={sample.original_pitch}, detune={sample.pitch_correction} cents")

    # Convert sample.

    conv = ConvertSession(opt, wav_path, brr_path, transpose=transpose)
    sample.sample_rate = coalesce(rate, conv.rate)

    if target_rate:
        # Based off WAV sample rate (conv.rate), not override!
        if ratio != 1:
            raise ValueError("Cannot specify both `target_rate` and `ratio`")
        ratio = Fraction(target_rate, conv.rate)
    else:
        ratio = Fraction(ratio)
    brr_result = conv.convert(
        ratio=ratio, loop=loop, truncate=truncate, volume=volume, decode=True
    )

    tune = tuning.brr_tune(sample, brr_result, tuning_, ncyc)
    print("tuning:", tune)

    return ConvertResult(tune)


"""
TODO --tuning-format=amk|c700
amk = write 2 bytes
c700 = write (root key: int, sampling rate: float)
"""


Folder = click.Path(exists=True, file_okay=False)


def ensure_positive(_ctx, param, value):
    if value is None:
        return value
    elif value >= 1:
        return value
    else:
        raise click.BadParameter("%s must be positive" % param)


def pathify(_ctx, _param, value):
    return Path(value)


def pathify_maybe(_ctx, _param, value):
    if value is None:
        return None
    return pathify(_ctx, _param, value)


@click.command()
@click.argument("wav_in", type=click.Path(exists=True), callback=pathify)
@click.argument("cfg_in", type=click.File("r"))
@click.argument("brr_out", type=click.Path(), callback=pathify)
@click.option("--sf2_in", "-S", type=click.File("rb"))
# @click.option("--yaml_out", "-Y", type=click.Path(), callback=pathify_maybe)
@click.option(
    *"decode_loops --decode-loops -D".split(), type=int, callback=ensure_positive
)
@click.option(*"verbose --verbose -v".split(), count=True)
def main(
    wav_in: Path,
    cfg_in: IO[str],
    brr_out: Path,
    sf2_in: Optional[IO[bytes]],
    # yaml_out: Optional[Path],
    decode_loops: Optional[int],
    verbose: int,
):
    # Begin wav2brr setup
    decode_loops = coalesce(decode_loops, 1)
    opt = CliOptions(verbose=verbose, decode_loops=decode_loops)

    if sf2_in is not None:
        sf2 = Sf2File(sf2_in)
        samples = sorted(
            sf2.samples[:-1], key=lambda s: s.name
        )  # type: List[Sf2Sample]
        name_to_sample = {
            sample.name: sample for sample in samples
        }  # type: Dict[str, Sf2Sample]
    else:
        name_to_sample = {}

    # Create .brr, decode to .wav
    result = convert_cfg(opt, wav_in, brr_out, cfg_in, name_to_sample)
    # TODO save tuning to file, if path or flag supplied (yaml_out)
