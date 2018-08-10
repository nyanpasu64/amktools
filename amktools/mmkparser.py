#!/usr/bin/env python3

# MMK Parser for AddMusicK
# Written by nobody1089
# Released under the WTFPL

import argparse
import os
import re
import sys
from contextlib import contextmanager
from fractions import Fraction
from io import StringIO
from pathlib import Path
from typing import Dict, List, Union, Pattern, Tuple, Callable, Optional

from dataclasses import dataclass, field
from ruamel.yaml import YAML

# We cannot identify instrument macros.
# The only way to fix that would be to expand macros, which would both complicate the program and
# make the generated source less human-readable.
from amktools.utils.math import ceildiv


class MMKError(ValueError):
    pass


def perr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


yaml = YAML(typ='safe')


def remove_ext(path):
    head = os.path.splitext(path)[0]
    return head


from amktools.common import TUNING_PATH, WAVETABLE_PATH
TXT_SUFFIX = '.txt'
RETURN_ERR = 1

def main(args: List[str]) -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
        description='Parse one or more MMK files to a single AddmusicK source file.',
        epilog='''Examples:
`mmk_parser file.mmk`                   outputs to file.txt
`mmk_parser file.mmk infile2.mmk`       outputs to file.txt
`mmk_parser file.mmk -o outfile.txt`    outputs to outfile.txt''')

    parser.add_argument('files', help='Input files, will be concatenated', nargs='+')
    parser.add_argument('-t', '--tuning',
                        help='Tuning file produced by wav2brr (defaults to {})'.format(TUNING_PATH))
    parser.add_argument('-w', '--wavetable',
                        help=f'Wavetable metadata produced by wavetable.to_brr (defaults to {WAVETABLE_PATH})')
    parser.add_argument('-o', '--outpath', help='Output path (if omitted)')
    args = parser.parse_args(args)

    # FILES
    inpaths = args.files
    first_path = inpaths[0]
    mmk_dir = Path(first_path).parent

    datas = []
    for _inpath in inpaths:
        with open(_inpath) as ifile:
            datas.append(ifile.read())
    datas.append('\n')
    in_str = '\n'.join(datas)


    # TUNING
    if 'tuning' in args:
        tuning_path = Path(args.tuning)
    else:
        tuning_path = mmk_dir / TUNING_PATH
    try:
        with open(tuning_path) as f:
            tuning = yaml.load(f)
        if type(tuning) != dict:
            perr('invalid tuning file {}, must be YAML key-value map'.format(
                tuning_path.resolve()))
            return RETURN_ERR
    except FileNotFoundError:
        tuning = None

    # WAVETABLE
    wavetable_path = vars(args).get('wavetable',
                                    mmk_dir / WAVETABLE_PATH)
    try:
        wavetable_path = Path(wavetable_path)
        wavetable = yaml.load(wavetable_path)
        wavetable = {k: WavetableMetadata(**meta) for k, meta in wavetable.items()}
            # type: Dict[str, WavetableMetadata]
    except FileNotFoundError:
        wavetable = None

    # OUT PATH
    if 'outpath' in args:
        outpath = args.outpath
    else:
        outpath = remove_ext(first_path) + TXT_SUFFIX

    for _inpath in inpaths:
        if Path(outpath).resolve() == Path(_inpath).resolve():
            perr('Error: Output file {} will overwrite an input file!'.format(outpath))
            if '.txt' in _inpath.lower():
                perr('Try renaming input files to .mmk')
            return RETURN_ERR

    # PARSE
    parser = MMKParser(in_str, tuning, wavetable)
    try:
        outstr = parser.parse()
    except MMKError as e:
        if str(e):
            perr('Error:', str(e))
        return RETURN_ERR

    with open(outpath, 'w') as ofile:
        ofile.write(outstr)

    return 0


def parse_int_round(instr):
    return int(parse_frac(instr))


def parse_int_hex(instr: str):
    if instr.startswith('$'):
        return int(instr[1:], 16)
    else:
        return int(instr, 0)


def parse_frac(infrac):
    if type(infrac) == str:
        slash_pos = infrac.find('/')
        if slash_pos != -1:
            num = infrac[:slash_pos]
            den = infrac[slash_pos + 1:]
            return Fraction(num) / Fraction(den)

    return Fraction(infrac)


def int2hex(in_frac):
    if not (isinstance(in_frac, int) and 0 <= in_frac <= 0xff):
        raise ValueError(f'Passed invalid value {in_frac} to int2hex')
    value = '$%02x' % int(in_frac)
    return value

to_hex = int2hex


OCTAVE = 12
notes = ['c', 'c+', 'd', 'd+', 'e', 'f', 'f+', 'g', 'g+', 'a', 'a+', 'b']
def format_note(midi: int):
    octave = (midi // OCTAVE) - 1
    note = notes[midi % OCTAVE]
    return f'o{octave}{note}'


QUARTER_TO_TICKS = 0x30


def vol_midi2smw(midi_vol):
    midi_vol = parse_frac(midi_vol)
    fractional = midi_vol / 127
    smw_vol = fractional * 255
    return round(smw_vol)


WHITESPACE = ' \t\n\r\x0b\f,:'
TERMINATORS = WHITESPACE + '"()[]'


def any_of(chars) -> Pattern:
    """ Compile chars into wildcard regex pattern.
    Match is 0 characters long and does not include char. """
    chars = ''.join(sorted(chars))
    regex = '(?=[{}])'.format(re.escape(chars))
    return re.compile(regex)


def none_of(chars) -> Pattern:
    """ Compile chars into negative-wildcard regex pattern.
    Match is 0 characters long and does not include non-matched char. """
    chars = ''.join(sorted(chars))
    regex = '(?=[^{}])'.format(re.escape(chars))
    return re.compile(regex)


def parse_time(word: str):
    return parse_frac(word) * QUARTER_TO_TICKS


@dataclass
class WavetableMetadata:
    nsamp: int
    ntick: int
    fps: float
    wave_sub: int   # Each wave is repeated `wave_sub` times.
    env_sub: int    # Each volume/frequency entry is repeated `env_sub` times.

    pitches: List[float]

    tuning: int = field(init=False)
    tuning_str: str = field(init=False)
    smp_idx: Optional[int] = None
    silent: bool = False
    def __post_init__(self):
        nsamp = self.nsamp
        if nsamp % 16:
            raise MMKError(f'cannot load sample with {nsamp} samples != n*16')
        self.tuning = nsamp // 16
        self.tuning_str = '$%02x $00' % self.tuning


class MMKParser:
    SHEBANG = '%mmk0.1'

    def __init__(
            self,
            in_str: str,
            tuning: Optional[Dict[str, str]],
            wavetable: Optional[Dict[str, WavetableMetadata]]
    ):
        # Input parameters
        self.in_str = in_str
        self.tuning = tuning
        self.wavetable = wavetable

        # Parser state
        self.orig_state = {
            'isvol': False, 'ispan': False, 'panscale': Fraction('5/64'), 'vmod': Fraction(1)}
        self.state = self.orig_state.copy()
        self.defines = {}  # type: Dict[str, str]

        # Wavetable parser state
        self.curr_chan: int = None
        self.smp_num = 0
        self.silent_idx: int = None

        # File IO
        self.pos = 0
        self.out = StringIO()


        # debug
        self._command = None
        self._begin_pos = 0

    # TODO MshFile object with read methods, line counts.

    def size(self):
        return len(self.in_str)

    # so I basically reimplemented the iterator protocol ad-hoc... except I can't use takewhile.
    # Iterators don't support peek(). https://pypi.org/project/more-itertools/ supports peek() like
    # my API.
    def peek(self):
        return self.in_str[self.pos]

    def is_eof(self):
        assert self.pos <= self.size()
        return self.pos >= self.size()

    def get_char(self) -> str:
        out = self.in_str[self.pos]
        self.pos += 1
        return out

    # **** I/O manipulation, AKA "wish I wrote a proper lexer/parser/output" ****

    @contextmanager
    def set_input(self, in_str: str):
        string = self.in_str
        begin = self.pos

        self.in_str = in_str
        self.pos = 0
        yield

        self.in_str = string
        self.pos = begin


    @contextmanager
    def end_at(self, sub):
        begin = self.pos
        in_str = self.get_until(sub, strict=False)
        end = self.pos

        with self.set_input(in_str):
            yield
            if begin + self.pos != end:
                raise Exception('Bounded parsing error, parsing ended at {} but region ends at {}'
                                .format(begin + self.pos, end))
        self.pos = end

    def until_comment(self):
        return self.end_at(any_of(';\n'))

    @contextmanager
    def capture(self) -> StringIO:
        orig = self.out

        self.out = StringIO()
        with self.out:
            yield self.out

        self.out = orig

    def parse_str(self, in_str: str):
        with self.set_input(in_str):
            self.parse()


    # **** Parsing ****
    def get_until(self, regex: Union[Pattern, str], strict) -> str:
        """
        Read until first regex match. Move pos after end of match (before lookahead).

        :param regex: Regex pattern terminating region.
        :param strict: If true, throws exception on failure. If false, returns in_str[pos:size()].
        :return: Text until regex match (optionally inclusive).
        """
        regex = re.compile(regex)
        match = regex.search(self.in_str, self.pos)

        if match:
            end = match.end()
            out_idx = match.start()
        elif not strict:
            end = self.size()
            out_idx = end
        else:
            raise MMKError('Unterminated region, missing "{}"'.format(regex.pattern))

        out = self.in_str[self.pos:out_idx]
        self.pos = end
        return out

    def get_chars(self, num) -> str:
        """ Gets the specified number of characters.
        :param num: Number of characters to skip.
        :return: String of characters
        """
        new = min(self.pos + num, self.size())
        skipped = self.in_str[self.pos:new]
        self.pos = new
        return skipped

    def skip_chars(self, num, keep: bool = True):
        skipped = self.get_chars(num)
        if keep:
            self.put(skipped)

    # High-level matching functions
    # Returns (parse, whitespace = skip_spaces())

    TERMINATORS_REGEX = any_of(TERMINATORS)  # 0-character match

    def get_word(self) -> Tuple[str, str]:
        """ Gets single word from file. If word begins with %, replaces with definition (used for parameters).
        Removes all leading spaces, but only trailing spaces up to the first \n.
        That helps preserve formatting.
        :return: (word, trailing whitespace)
        """

        self.skip_spaces(False)

        word = self.get_until(self.TERMINATORS_REGEX, strict=False)
        if not word:
            raise Exception('Tried to get word where none exists (invalid command or missing arguments?)')
        whitespace = self.get_spaces(exclude='\n')

        if word.startswith('%'):
            word = self.defines.get(word[1:], word)     # dead code?
        return word, whitespace

    def get_phrase(self, n: int) -> List[str]:
        """ Gets n words, plus trailing whitespace. """
        if n <= 0:
            raise ValueError('invalid n={} < 0'.format(repr(n)))

        words = []
        whitespace = None
        for i in range(n):
            word, whitespace = self.get_word()
            words.append(word)

        words.append(whitespace)
        return words


    def get_spaces(self, exclude='') -> str:
        whitespace = set(WHITESPACE) - set(exclude)
        not_whitespace = none_of(whitespace)    # 0-character match
        skipped = self.get_until(not_whitespace, strict=False)
        return skipped

    def skip_spaces(self, keep: bool, exclude=''):
        skipped = self.get_spaces(exclude)
        if keep:
            self.put(skipped)


    def get_quoted(self):
        """
        :return: contents of quotes
        """
        if self.get_char() != '"':
            raise MMKError('string does not start with "')
        quoted = self.get_until(r'["]', strict=True)
        whitespace = self.get_spaces(exclude='\n')
        return quoted, whitespace

    def get_line(self):
        return self.get_until(any_of('\n'), strict=False)

    # Returns parse (doesn't fetch trailing whitespace)
    def get_int(self, maybe=False) -> Optional[int]:
        buffer = ''
        while self.peek().isdigit():
            buffer += self.get_char()

        if not buffer and maybe:
            return None
        return parse_int_round(buffer)

    def put(self, pstr):
        self.out.write(pstr)

    def put_hex(self, *nums):
        not_first = False
        for num in nums:
            if not_first:
                self.put(' ')

            self.put(to_hex(num))
            not_first = True
        self.put('  ')

    # Begin parsing functions!
    def parse_define(self, command_case, whitespace):
        """ TODO Parse literal define, passthrough. """
        if command_case in self.defines:
            self.put(self.defines[command_case] + whitespace)
            return True
        return False

    # **** Transpose ****

    def parse_transpose(self) -> None:
        transpose_str, whitespace = self.get_phrase(1)
        transpose = parse_int_hex(transpose_str)

        if transpose not in range(-0x80, 0x80):
            raise MMKError('invalid transpose {}'.format(transpose_str))

        transpose_hex = int2hex(transpose & 0xff)
        self.put('$FA $02 {}'.format(transpose_hex))
        self.put(whitespace)

    # **** volume ****

    def calc_vol(self, in_vol):
        vol = parse_frac(in_vol)
        vol *= self.state['vmod']

        if self.state['isvol']:
            vol *= 2
        return str(round(vol))

    def parse_vol(self):
        self.skip_chars(1, keep=False)
        orig_vol = self.get_int(maybe=True)
        if orig_vol is None:
            return

        self.state['vol'] = self.calc_vol(orig_vol)
        self.put('v' + self.state['vol'])

    def parse_vol_hex(self, arg):
        # This both returns the volume and modifies state.
        # Time to throw away state?
        new_vol = self.state['vol'] = self.calc_vol(arg)    # type: str
        hex_vol = int2hex(new_vol)
        return hex_vol

    def parse_vbend(self, time, vol, whitespace):
        # Takes a fraction of a quarter note as input.
        # Converts to ticks.
        time_hex = int2hex(parse_time(time))
        vol_hex = self.parse_vol_hex(vol)

        self.put('$E8 {} {}{}'.format(time_hex, vol_hex, whitespace))

    # **** pan ****

    def calc_pan(self, orig_pan):
        # Convert panning
        if self.state['ispan']:
            zeroed_pan = parse_frac(orig_pan) - 64
            scaled_pan = zeroed_pan * self.state['panscale']
            return str(round(scaled_pan + 10))
        else:
            return str(orig_pan)

    def parse_pan(self):
        self.skip_chars(1, keep=False)
        orig_pan = self.get_int(maybe=True)
        if orig_pan is None:
            return

        self.state['pan'] = self.calc_pan(orig_pan)
        # Pass the command through.
        self.put('y' + self.state['pan'])

    def parse_ybend(self, duration, pan):
        duration_hex = int2hex(parse_time(duration))
        self.state['pan'] = self.calc_pan(pan)
        pan_hex = int2hex(self.state['pan'])

        self.put('$DC {} {}'.format(duration_hex, pan_hex))

    # **** meh ****

    def skip_until(self, end):
        # FIXME deprecated
        in_str = self.in_str
        self.skip_chars(1, keep=True)
        end_pos = in_str.find(end, self.pos)
        if end_pos == -1:
            end_pos = self.size()

        # The delimiter is skipped as well.
        # If end_pos == self.len(), skip_chars handles the OOB case by not reading the extra char.
        self.skip_chars(end_pos - self.pos + 1, keep=True)

        return self.in_str[end_pos]

    def parse_comment(self):
        self.skip_until('\n')

    # Multi-word parsing

    def parse_pbend(self, delay, time, note, whitespace):
        # Takes a fraction of a quarter note as input.
        # Converts to ticks.
        delay_hex = int2hex(parse_time(delay))
        time_hex = int2hex(parse_time(time))

        self.put('$DD {} {} {}{}'.format(delay_hex, time_hex, note, whitespace))

    # **** oscillatory effects ****

    def parse_vib(self, delay, frequency, amplitude, whitespace):
        delay_hex = int2hex(parse_time(delay))
        freq_hex = int2hex(parse_frac(frequency))

        self.put('$DE {} {} {}{}'.format(delay_hex, freq_hex, amplitude, whitespace))

    def parse_trem(self, delay, frequency, amplitude, whitespace):
        delay_hex = int2hex(parse_time(delay))
        freq_hex = int2hex(parse_frac(frequency))

        self.put('$E5 {} {} {}{}'.format(delay_hex, freq_hex, amplitude, whitespace))

    # **** envelope effects ****

    _GAINS = [
        # curve, begin, max_rate
        ['direct', 'set', 0x00],
        ['down', 0x80],
        ['exp', 0xa0],
        ['up', 0xc0],
        ['bent', 0xe0],
        [None, 0x100],
    ]

    for i in range(len(_GAINS) - 1):
        _GAINS[i].append(_GAINS[i + 1][-1] - _GAINS[i][-1])
    _GAINS = _GAINS[:-1]

    def parse_gain(self, *, instr):
        # Look for a matching GAIN value, ensure the input rate lies in-bounds,
        # then write a hex command.
        curve, rate, whitespace = self.get_phrase(2)

        if instr:
            prefix = '$00 $00'
        else:
            prefix = '$FA $01'

        raw_rate = rate
        rate = parse_int_hex(rate)
        for *curves, begin, max_rate in self._GAINS:
            if curve in curves:
                rate = self._index_check(curve, rate, max_rate)
                self.put('%s %s%s' % (prefix, int2hex(begin + rate), whitespace))
                return

        perr('Invalid gain %s, options are:' % repr(curve))
        for curve, _, max_rate in self._GAINS:
            perr('%s (rate < %s)' % (curve, hex(max_rate)))
        raise MMKError

    def parse_adsr(self, instr: bool):
        """
        Parse ADSR command.
        attack: Attack speed (0-15)
        decay: Decay speed (0-7)
        sustain: Sustain volume (0-7)
        release: Release speed (0-31)

        :param instr: Whether ADSR command occurs in instrument definition (or MML command)
        """
        attack, decay, sustain, release, whitespace = self.get_phrase(4)
        if sustain.startswith('full'):
            sustain = '7'

        attack = parse_int_hex(attack)
        decay = parse_int_hex(decay)
        sustain = parse_int_hex(sustain)
        release = parse_int_hex(release)

        attack = self._index_check('attack', attack, 0x10)
        decay = self._index_check('decay', decay, 0x08)
        sustain = self._index_check('sustain', sustain, 0x08)
        release = self._index_check('release', release, 0x20)

        a = 0x10 * decay + attack
        b = 0x20 * sustain + release

        if instr:
            a += 0x80
            fmt = '{} {} $A0'
        else:
            fmt = '$ED {} {}'
        self.put(fmt.format(int2hex(a), int2hex(b)))
        self.put(whitespace)

    @staticmethod
    def _index_check(caption, val, end):
        if val < 0:
            val += end
        if val not in range(end):
            raise MMKError('Invalid ADSR/gain {} {} (must be < {})'.format(caption, val, end))
        return val

    # **** event handler callbacks ****
    event_map = {
        'clear': 0,

        'keyon': -1,
        'kon': -1,
        'begin': -1,
        'start': -1,

        'after': 1,     # after keyon

        'before': 2,    # before keyoff

        'keyoff': 3,
        'koff': 3,
        'kof': 3,
        'end': 3,

        'now': 4
    }
    def parse_callback(self):
        expr = self.get_until(any_of(')'), strict=True)
        args = [word.strip() for word in expr.split()]

        # if len(args) < 1:
        #     raise MMKError(
        #         f"Invalid callback (!{expr}), must have (!callback)[] or (!callback, event)")

        if len(args) < 2:
            # Callback definition (!n)
            self.put(expr)
            return

        callback_num = args[0]
        event = args[1]
        event_num = self.event_map[event]

        if event in ['after', 'before']:
            time = args[2]
            if len(args) != 3:
                raise MMKError(
                    f"Invalid event binding (!{expr}), must have duration (measure/$x)")
            self.put('{}, {}, {}'.format(callback_num, event_num, time))
        else:
            self.put('{}, {}'.format(callback_num, event_num))

    # **** #instruments ****

    def parse_tune(self):
        self.smp_num += 1
        # "test.brr" $ad $sr $gain $tune $tune

        brr, whitespace = self.get_quoted()

        if self.tuning is None:
            perr('Cannot use %tune without a tuning file')
            raise MMKError

        tuning = self.tuning[brr]

        self.put('"{}"{}'.format(brr, whitespace))
        with self.end_at(any_of(';\n')):
            self.parse_instruments()  # adsr+gain
        self.put(' {}'.format(tuning))


    # **** Wavetable sweeps ****

    def parse_smp(self):
        self.smp_num += 1

    def parse_silent(self):
        self.silent_idx = self.smp_num
        self.smp_num += 1

    def parse_group(self):
        self.smp_num += self.get_int()

    def parse_wave_group(self, is_instruments: bool):
        name, whitespace = self.get_quoted()
        ntick_playback = None

        if not is_instruments:
            ntick_playback = self.get_int()     # Only load the first N ticks
            whitespace = self.get_spaces(exclude='\n')
            # ntick_playback, whitespace = self.get_word()     # The sweep lasts for N ticks
            # ntick_playback = int(parse_time(ntick_playback))

        meta = self.wavetable[name]
        waves = self._get_waves_in_group(name, ntick_playback)

        with self.capture() as output, self.until_comment():
            if is_instruments:
                self.parse_instruments()
                self.put(' ' + meta.tuning_str)
                # *ugh* the instrument's tuning value is basically unused
            else:
                self.put(self.get_line())

            after = output.getvalue()
            if not is_instruments:  # If samples
                args = after.split()
                after = after[len(after.rstrip()):]     # Only keep whitespace

                # print(name, args)
                for arg in args:
                    if arg in ['silent']:
                        setattr(meta, arg, True)
                    else:
                        raise MMKError(f'Invalid #samples{{%wave_group}} argument {arg}')
        comments = self.get_line()
        self.skip_chars(1, keep=False)  # remove trailing newline

        for wave in waves:
            # eh, missing indentation. who cares.
            self.put(f'"{wave}"{whitespace}{after}{comments}\n')
            comments = ''

        if not is_instruments:  # FIXME
            meta.smp_idx = self.smp_num
            self.smp_num += len(waves)

    WAVE_GROUP_TEMPLATE = '{}-{:03}.brr'

    def _get_waves_in_group(self, name: str, ntick_playback: int) -> List[str]:
        """ Returns a list of N BRR wave names. """
        # if name in self.wave_groups:
        #     return self.wave_groups[name]

        if self.wavetable is None:
            raise MMKError('cannot load wavetables, missing wavetable.yaml')

        meta = self.wavetable[name]

        if ntick_playback is not None:
            meta.ntick = min(meta.ntick, ntick_playback)

        nwave = ceildiv(meta.ntick, meta.wave_sub)
        wave_names = [self.WAVE_GROUP_TEMPLATE.format(name, i) for i in range(nwave)]
        return wave_names

    DETUNE = 0xEE
    LEGATO = '$F4 $01  '
    def parse_wave_sweep(self):
        """ Print a wavetable sweep. """
        name, _ = self.get_quoted()
        ntick_note = self.get_int()     # The sweep lasts for N ticks
        meta = self.wavetable[name]

        # Load silent instrument with proper tuning
        self.parse_str('%adsr -3,-1,full,0  ')
        if self.silent_idx is None:
            raise MMKError('cannot %wave_sweep without silent sample defined')
        self.put_hex(0xf3, self.silent_idx, meta.tuning)
        self.put('  ')
        self.put(self.LEGATO)   # Legato glues right+2, and unglues left+right.

        # Pitch tracking
        midi = 0   # MIDI

        # Each note follows a pitch/wave event. It is printed with the
        # proper duration, when the next pitch/wave event begins.
        prev_tick = 0
        def print_note():
            nonlocal prev_tick
            if tick > prev_tick:
                self.put(f'{format_note(midi)}={tick - prev_tick}  ')
                prev_tick = tick

        ntick_instr = min(ntick_note, meta.ntick)
        wave_idx = 0
        for tick in range(ntick_instr):
            # Wave envelope
            if tick % meta.wave_sub == 0:
                wave_idx = tick // meta.wave_sub
                print_note()
                # Print wave
                self._put_load_sample(meta.smp_idx + wave_idx)

            # Pitch envelope
            if tick % meta.env_sub == 0:
                env_idx = tick // meta.env_sub
                print_note()

                # Print pitch
                _pitch = meta.pitches[env_idx]
                midi = int(_pitch)
                detune = _pitch - midi
                self.put_hex(self.DETUNE, int(detune * 256))
                # midi is used by print_note()

        if meta.silent:
            tick = ntick_instr
            print_note()

            # GAIN starts when the right note starts.
            self.parse_str('%gain down $18  ')
            tick = ntick_note
            print_note()
        else:
            tick = ntick_note
            print_note()

        self.put(self.LEGATO)   # Legato deactivates immediately.
        self.put_hex(self.DETUNE, 0)

    def _get_wave_reg(self):
        return 0x10 * self.curr_chan + 0x04

    _REG = 0xF6
    def _put_load_sample(self, smp_idx: int):
        self.put_hex(self._REG, self._get_wave_reg(), smp_idx)

    # self.state:
    # PAN, VOL, INSTR: str (Remove segments?)
    # PANSCALE: Fraction (5/64)
    # ISVOL, ISPAN: bool

    def parse(self) -> str:
        # For exception debug
        try:
            # Remove the header. TODO increment pos instead.
            if self.in_str.startswith(self.SHEBANG):
                self.in_str = self.in_str[len(self.SHEBANG):].lstrip()  # type: str

            while self.pos < self.size():
                # Yeah, simpler this way. But could hide bugs/inconsistencies.
                self.skip_spaces(True)

                if self.pos == self.size():
                    break
                    # FIXME COMMENT Only whitespace left, means already printed, nothing more to do
                self._begin_pos = self.pos
                char = self.peek()

                # TODO refactor to elif (so you can't forget continue)
                # and make functions get their own args.

                # Parse the no-argument default commands.
                if char == 'v':
                    self.parse_vol()
                    continue

                if char == 'y':
                    self.parse_pan()
                    continue

                if char == ';':
                    self.parse_comment()
                    continue

                if char == '#':  # instruments{}
                    self.skip_chars(1, keep=True)
                    self.skip_spaces(True)

                    def branch(keyword: str, method: Callable):
                        if self.in_str.startswith(keyword, self.pos):
                            self.skip_until('{')
                            self.skip_chars(1, keep=True)
                            method()

                    branch('samples', self.parse_samples)
                    branch('instruments', self.parse_instruments)
                    if self.peek().isnumeric():
                        chan = self.get_char()
                        self.curr_chan = int(chan)
                        self.put(chan)

                    continue

                if char == '(':
                    self.skip_chars(1, keep=True)
                    if self.peek() == '!':
                        self.skip_chars(1, keep=True)
                        self.parse_callback()
                    continue

                # Begin custom commands.
                if char == '%':
                    self.skip_chars(1, keep=False)

                    # NO ARGUMENTS
                    command_case, whitespace = self.get_word()
                    command = command_case.lower()
                    self._command = command

                    if self.parse_define(command_case, whitespace):
                        continue

                    if command == 'mmk0.1':
                        raise Exception("this shouldn't happen")

                    if command == 'define':
                        key = self.get_word()[0]
                        value = self.get_line()
                        self.defines[key] = value
                        continue

                    if command == 'reset':
                        self.state = self.orig_state.copy()
                        continue

                    if command == 'passthrough':
                        self.state['currseg'] = -1
                        continue

                    if command == 'isvol':
                        self.state['isvol'] = True
                        continue

                    if command == 'ispan':
                        self.state['ispan'] = True
                        continue

                    if command == 'notvol':
                        self.state['isvol'] = False
                        continue

                    if command == 'notpan':
                        self.state['ispan'] = False
                        continue

                    # N ARGUMENTS

                    if command in ['t', 'transpose']:
                        self.parse_transpose()
                        continue

                    if command == 'adsr':
                        self.parse_adsr(instr=False)
                        continue

                    if command == 'gain':
                        self.parse_gain(instr=False)
                        continue

                    # Wavetable sweep
                    if command == 'wave_sweep':
                        self.parse_wave_sweep()
                        continue

                    # ONE ARGUMENT
                    arg, whitespace = self.get_word()

                    if command == 'vmod':
                        self.state['vmod'] = parse_frac(arg)
                        continue

                    # 2 ARGUMENTS
                    arg2, whitespace = self.get_word()
                    if command in ['vbend', 'vb']:
                        self.parse_vbend(arg, arg2, whitespace)
                        continue

                    if command in ['ybend', 'yb']:
                        self.parse_ybend(duration=arg, pan=arg2)
                        self.put(whitespace)
                        continue

                    # 3 ARGUMENTS
                    arg3, whitespace = self.get_word()

                    if command == 'vib':
                        self.parse_vib(arg, arg2, arg3, whitespace)
                        continue

                    if command == 'trem':
                        self.parse_trem(arg, arg2, arg3, whitespace)
                        continue

                    if command in ['pbend', 'pb']:
                        self.parse_pbend(arg, arg2, arg3, whitespace)
                        continue

                    # INVALID COMMAND
                    raise MMKError('Invalid command ' + command)
                else:
                    self.skip_chars(1, keep=True)
                    self.skip_spaces(True)

            return self.out.getvalue().strip() + '\n'
        except Exception:
            # Seek at least 100 characters back
            begin_pos = self._begin_pos
            idx = begin_pos
            for i in range(3):
                idx = self.in_str.rfind('\n', 0, idx)
                if idx == -1:
                    break
                if begin_pos - idx >= 100:
                    break
            idx += 1

            if self._command is None:
                last = 'None'
            else:
                last = '%' + self._command
            perr()
            perr('#### MMK parsing error ####')
            perr('  Last command: ' + last)
            perr('  Context:')
            perr(self.in_str[idx:begin_pos] + '...\n')

            raise  # main() eats MMKError to avoid visual noise

    # noinspection PyMethodParameters
    def _brace_parser_factory(mapping: Dict[str, Callable[['MMKParser'], None]]) \
            -> Callable:
        def _parse(self: 'MMKParser'):
            """
            Parses #instruments{...} blocks. Eats trailing close-brace.
            Also used for parsing quoted BRR filenames within #instruments.
            """
            close = '}'

            while self.pos < self.size():
                # pos = self.pos
                self.skip_spaces(True, exclude=close)
                self._begin_pos = self.pos
                # assert pos == self.pos
                char = self.peek()

                if char in close:
                    self.skip_chars(1, True)  # {}, ""
                    self.skip_spaces(True, exclude='\n')
                    return

                if char == '%':
                    self.skip_chars(1, False)

                    command_case, whitespace = self.get_word()
                    command = command_case.lower()
                    self._command = command

                    # **** Parse defines ****
                    if self.parse_define(command_case, whitespace):
                        pass
                    # **** Parse commands ****
                    elif command in mapping:
                        mapping[command](self)
                    else:
                        perr(mapping.keys())
                        raise MMKError('Invalid command ' + command)
                else:
                    self.skip_chars(1, keep=True)
                    self.skip_spaces(True)
        return _parse


    # noinspection PyArgumentList
    parse_instruments = _brace_parser_factory({
        'group': lambda self: self.parse_group(),
        'tune': lambda self: self.parse_tune(),
        'gain': lambda self: self.parse_gain(instr=True),
        'adsr': lambda self: self.parse_adsr(instr=True),
        'wave_group': lambda self: self.parse_wave_group(is_instruments=True),
    })

    # noinspection PyArgumentList
    parse_samples = _brace_parser_factory({
        'smp': lambda self: self.parse_smp(),
        'silent': lambda self: self.parse_silent(),
        'wave_group': lambda self: self.parse_wave_group(is_instruments=False),
    })

if __name__ == '__main__':
    ret = main(sys.argv[1:])
    exit(ret)
