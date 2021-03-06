#!/usr/bin/env python3

# MMK Parser for AddMusicK
# Written by nobody1089
# Released under the WTFPL

import argparse
import copy
import heapq
import itertools
import math
import numbers
import os
import re
import sys
from abc import abstractmethod, ABC
from contextlib import contextmanager
from fractions import Fraction
from io import StringIO
from pathlib import Path
from typing import Dict, List, Union, Pattern, Tuple, Callable, Optional, ClassVar, \
    Iterator, TypeVar, Iterable

import dataclasses
import pygtrie
from dataclasses import dataclass, field
from more_itertools import peekable, split_before
from ruamel.yaml import YAML

# We cannot identify instrument macros.
# The only way to fix that would be to expand macros, which would both complicate the program and
# make the generated source less human-readable.
from amktools.util import ceildiv, coalesce
from amktools.utils.parsing import safe_eval
from amktools.utils.substring_trie import StringSlice


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
        wavetable = {k: WavetableMetadata(name=k, **meta) for k, meta in wavetable.items()}
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


def try_int(s):
    try:
        return int(s, 10)
    except ValueError:
        return None


def parse_hex_only(in_str: str):
    if in_str.startswith('$'):
        return int(in_str[1:], 16)
    else:
        hex_value = int(in_str, 16)
        int_value = try_int(in_str)
        if int_value is not None and int_value != hex_value:
            raise MMKError(
                f'Ambiguous value {in_str} in hexadecimal-only field (try prefixing $)')
        return hex_value


def parse_frac(infrac):
    if type(infrac) == str:
        slash_pos = infrac.find('/')
        if slash_pos != -1:
            num = infrac[:slash_pos]
            den = infrac[slash_pos + 1:]
            return Fraction(num) / Fraction(den)

    return Fraction(infrac)


def to_hex(in_frac):
    in_frac = int(in_frac)
    if not (-0x80 <= in_frac < 0x100):
        raise ValueError(f'Passed invalid {type(in_frac)} {in_frac} to int2hex')
    value = '$%02x' % (in_frac % 0x100)
    return value


def parse_wave_range(sweep_str: str, nwave: int) -> range:
    """ Parse range of wave sample indices. """
    error = MMKError(f'wave range {sweep_str} invalid, must be [x,y] or [x,y)')

    begin_end = sweep_str.split(',')
    if len(begin_end) != 2:
        raise error
    begin_str, end_str = begin_end

    # Either sample index, or fraction {1., .5} of nwave.
    # TODO move bounds checking to SweepEvent
    def _parse_ratio_or_int(s: str) -> int:
        if '.' in s:
            return round(nwave * float(s))
        else:
            return parse_int_hex(s)

    # [Begin interval
    open_paren = begin_str[0]
    if open_paren not in '[(':
        raise error
    begin_idx = _parse_ratio_or_int(begin_str[1:])

    # End interval)
    close_paren = end_str[-1]
    if close_paren not in '])':
        raise error
    end_idx = _parse_ratio_or_int((end_str[:-1]))

    # Python range() defaults to [x..y-1]. We can turn it into [x+1..y].
    delta = int(math.copysign(1, end_idx - begin_idx))
    if open_paren == '(':
        begin_idx += delta
    if close_paren == ']':
        end_idx += delta

    return range(begin_idx, end_idx, delta)


OCTAVE = 12
note_names = ['c', 'c+', 'd', 'd+', 'e', 'f', 'f+', 'g', 'g+', 'a', 'a+', 'b']
def format_note(midi: int):
    octave = (midi // OCTAVE) - 1
    note = note_names[midi % OCTAVE]
    return f'o{octave}{note}'

note2pitch = {note: idx for idx, note in enumerate(note_names)}
accidental2pitch = {'+': 1, '-': -1}


TICKS_PER_BEAT = 0x30
TICKS_PER_MEASURE = 4 * TICKS_PER_BEAT


def vol_midi2smw(midi_vol):
    midi_vol = parse_frac(midi_vol)
    fractional = midi_vol / 127
    smw_vol = fractional * 255
    return round(smw_vol)


WHITESPACE = ' \t\n\r\x0b\f,'
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


@dataclass
class WavetableMetadata:
    nwave: int = field(init=False)
    nsamp: int
    ntick: int
    fps: float      # Unused. %wave_sweep (constant rate) assumes fps = ticks/second.
    wave_sub: int   # Each wave is repeated `wave_sub` times.
    env_sub: int    # Each volume/frequency entry is repeated `env_sub` times.

    root_pitch: int
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

    name: str = None


class Stream:
    # Idea: Stream object with read methods, get_word, etc.
    # And external parse_... functions.

    SHEBANG = '%mmk0.1'

    def __init__(self, in_str: str, defines: Dict[str, str], remove_shebang=False):
        """
        Construct an input Stream.
        :param in_str: string
        :param defines: Passed by reference.
        :param remove_shebang: Only True on first Stream created
            (not on #instruments{...}).
        """
        self.in_str = in_str
        self.defines = defines
        self.pos = 0

        if remove_shebang:
            if self.in_str.startswith(self.SHEBANG):
                self.in_str = self.in_str[len(self.SHEBANG):].lstrip()

    def size(self):
        return len(self.in_str)

    # so I basically reimplemented the iterator protocol ad-hoc... except I can't use takewhile.
    # Iterators don't support peek(). https://pypi.org/project/more-itertools/ supports peek() like
    # my API.
    def peek(self) -> str:
        if self.is_eof():
            # Return an "invalid Unicode character". Will it hide bugs?
            return '\uFFFF'
        return self.in_str[self.pos]

    def peek_equals(self, keyword: str):
        return self.in_str.startswith(keyword, self.pos)


    def is_eof(self):
        assert self.pos <= self.size()
        return self.pos >= self.size()  # TODO ==

    def get_char(self) -> str:
        out = self.in_str[self.pos]
        self.pos += 1
        return out

    # **** Parsing ****
    def get_until(self, regex: Union[Pattern, str], strict) -> str:
        """
        Read until first regex match. Move pos after end of match (before lookahead).

        :param regex: Regex pattern terminating region.
        :param strict: If true, throws exception on failure. If false, returns in_str[pos:size()].
        :return: Text until (not including) regex match.
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

    def get_chars(self, num: int) -> str:
        """ Gets the specified number of characters.
        :param num: Number of characters to skip.
        :return: String of characters
        """
        new = min(self.pos + num, self.size())
        skipped = self.in_str[self.pos:new]
        self.pos = new
        return skipped

    def skip_chars(self, num, put: Callable = None):
        skipped = self.get_chars(num)
        if put:
            put(skipped)

    def skip_until(self, end: str, put: Callable):
        # FIXME deprecated
        in_str = self.in_str
        self.skip_chars(1, put)
        end_pos = in_str.find(end, self.pos)
        if end_pos == -1:
            end_pos = self.size()

        # The delimiter is skipped as well.
        # If end_pos == self.len(), skip_chars handles the OOB case by not reading the extra char.
        self.skip_chars(end_pos - self.pos + 1, put)

        return self.in_str[end_pos]

    # High-level matching functions
    # Returns (parse, whitespace = skip_spaces())

    TERMINATORS_REGEX = any_of(TERMINATORS)  # 0-character match

    def get_word(self, terminators=None) -> Tuple[str, str]:
        """ Gets single word from file. If word begins with %, replaces with definition (used for parameters).
        Removes all leading spaces, but only trailing spaces up to the first \n.
        That helps preserve formatting.
        :param terminators: Custom set of characters to include
        :return: (word, trailing whitespace)
        """

        self.skip_spaces()

        if terminators:
            regex = re.compile(any_of(terminators))
        else:
            regex = self.TERMINATORS_REGEX

        word = self.get_until(regex, strict=False)
        if not word:
            raise ValueError('Tried to get word where none exists (invalid command or missing arguments?)')
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


    def get_spaces(self, exclude: Iterable[str] = '') -> str:
        whitespace = set(WHITESPACE) - set(exclude)
        not_whitespace = none_of(whitespace)    # 0-character match
        skipped = self.get_until(not_whitespace, strict=False)
        return skipped

    def skip_spaces(self, put: Callable = None, exclude: Iterable[str] = ''):
        skipped = self.get_spaces(exclude)
        if put:
            put(skipped)

    def get_line_spaces(self):
        # TODO use function more
        return self.get_spaces(exclude='\n')


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
        # TODO add "put" parameter
        return self.get_until(any_of('\n'), strict=False)

    # Returns parse (doesn't fetch trailing whitespace)
    def get_int(self, maybe=False) -> Optional[int]:
        buffer = ''
        while self.peek().isdigit():
            buffer += self.get_char()

        if not buffer:
            if maybe:
                return None
            else:
                raise MMKError('Integer expected, but no digits to parse')
        return parse_int_round(buffer)

    def get_time(self) -> Tuple[Optional[int], str]:
        """ Obtains time and fetches trailing whitespace.

        Returns (nticks, whitespace). """
        dur = self._get_time()
        whitespace = self.get_spaces(exclude='\n')
        return dur, whitespace


    def _get_time(self) -> Optional[int]:
        """ Obtains time without getting trailing whitespace.

        Returns nticks. """
        first = self.peek()

        if first == '=':
            # =48
            self.skip_chars(1)
            return self.get_int()

        is_numerator = first.isnumeric()
        is_reciprocal = (first == '/')

        if not (is_numerator or is_reciprocal):
            # no duration specified
            return None

        if is_numerator:
            # 1, 1/48
            num = self.get_int()
        else:
            # /48
            num = 1

        if self.peek() == '/':
            # 1/48. /48
            self.skip_chars(1)
            den = self.get_int()
        else:
            # 1
            den = 1

        dur = Fraction(num, den) * TICKS_PER_BEAT
        if int(dur) != dur:
            raise MMKError(
                f'Invalid duration {Fraction(num/den)}, must be multiple of 1/48')

        return int(dur)


# Takes total duration and returns note duration.
NoteLenCalc = Callable[[int], int]
NoteLenFactory = Callable[[int], NoteLenCalc]


def release_early(dur: int) -> NoteLenCalc:
    def _release(note_len: int) -> int:
        return note_len - dur
    return _release


def staccato(dur: int) -> NoteLenCalc:
    # Disable staccato.
    if dur == 0:
        return release_early(0)

    # Return dur-tick-long staccato.
    def _staccato(note_len: int) -> int:
        return min(note_len, dur)
    return _staccato


@dataclass
class MMKState:
    isvol: bool = False
    ispan: bool = False
    is_notelen: bool = False
    panscale: Fraction = Fraction('5/64')
    vmod: Fraction = Fraction(1)

    # Note staccato and early release
    default_note_len: int = None
    # staccato: int = 0
    note_len_calc: NoteLenCalc = release_early(0)

    v: Optional[str] = None
    y: str = '10'

    keys: ClassVar = ['v', 'y']


# TODO move parsers from methods to functions


NOTES_WITH_DURATION = frozenset('abcdefg^rl')
RELEASE_CHAR = '~'
STACCATO_CHAR = '.'  # Dots have a different meaning from normal MML.

class MMKParser:
    FIRST_INSTRUMENT = 30

    def __init__(
            self,
            in_str: str,
            tuning: Optional[Dict[str, str]],
            wavetable: Optional[Dict[str, WavetableMetadata]] = None
    ):
        # Input parameters
        self.tuning = tuning
        self.wavetable = wavetable

        # Parser state
        self.orig_state = MMKState()
        self.state = copy.copy(self.orig_state)
        self.defines = dict(
            viboff='$DF',
            tremoff='$E5 $00 $00 $00',
            slur='$F4 $01',
            legato='$F4 $01',
            light='$F4 $02',
            restore_instr='$F4 $09'
        )  # type: Dict[str, str]
        self.amk_keys = pygtrie.CharTrie()

        # Wavetable parser state
        self.curr_chan: int = None
        self.smp_num = 0
        self.instr_num = self.FIRST_INSTRUMENT
        self.silent_idx: int = None

        # File IO
        self.stream = Stream(in_str, self.defines, remove_shebang=True)
        self.out = StringIO()

        # To print exception location
        self._command = None
        self._begin_pos = 0

    # **** I/O manipulation, AKA "wish I wrote a proper lexer/parser/output" ****

    @contextmanager
    def set_input(self, in_str: str):
        """ Temporarily replaces self.stream with new string.
        Idea: Maybe parser functions should take a stream parameter?
        """
        stream = self.stream
        self.stream = Stream(in_str, self.defines)
        try:
            yield
        finally:
            self.stream = stream


    @contextmanager
    def end_at(self, end_regex: Pattern):
        """ Temporarily replaces self.stream with truncated version. """
        in_str = self.stream.get_until(end_regex, strict=False)
        with self.set_input(in_str):
            yield
            if not self.stream.is_eof():
                raise Exception(
                    'Bounded parsing error, parsing ended at {} but region ends at {}'
                        .format(self.stream.pos, len(in_str)))


    def until_comment(self):
        return self.end_at(any_of(';\n'))

    @contextmanager
    def capture(self) -> StringIO:
        orig = self.out

        self.out = StringIO()
        with self.out:
            try:
                yield self.out
            finally:
                self.out = orig

    def parse_str(self, in_str: str):
        with self.set_input(in_str):
            self.parse()

    # Writing strings

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
    def parse_amk_replace(self):
        assert self.stream.get_char() == '"'

        before = self.stream.get_until('=', strict=True)
        after = self.stream.get_until('"', strict=True)
        self.amk_keys[before.strip()] = True

        self.put('"{}='.format(before))
        self.parse_str(after)
        self.put('"')

    def subst_define(self, command_case, whitespace):
        """ TODO Parse literal define, passthrough. """
        if command_case in self.defines:
            self.put(self.defines[command_case] + whitespace)
            return True
        return False

    # Save/restore state

    def parse_save(self):
        assert self.state is not self.orig_state
        self.orig_state = copy.copy(self.state)
        assert self.state is not self.orig_state

    def parse_restore(self):
        assert self.state is not self.orig_state

        for key in MMKState.keys:
            old = getattr(self.orig_state, key)
            new = getattr(self.state, key)
            if old != new:
                self.put(key + old)
        self.state = copy.copy(self.orig_state)

        assert self.state is not self.orig_state

    # **** Numerator-fraction note lengths ****

    WORD_TO_BOOL = dict(
        on=True,
        off=False,
        true=True,
        false=False
    )

    def parse_toggle_notelen(self):
        word, _ = self.stream.get_word()
        try:
            state = self.WORD_TO_BOOL[word]
        except KeyError:
            raise MMKError(
                f"invalid %notelen value {word}, expected {self.WORD_TO_BOOL.keys()}"
            )

        self.state.is_notelen = state

    def parse_note(self, allowed_notes=NOTES_WITH_DURATION):
        """ Parse a fractional note, and write a tick count. """
        note_str = self.stream.get_char()
        if note_str not in allowed_notes:
            allowed_str = ''.join(sorted(allowed_notes))
            raise MMKError(f'Invalid note name {note_str} not in {allowed_str}')
        if self.stream.peek() in '+-':
            note_str += self.stream.get_char()

        # If no duration supplied, nticks is None.
        nticks, whitespace = self.stream.get_time()

        if note_str == 'l':
            if nticks is None:
                raise MMKError('Cannot use lxx command without duration')
            self.state.default_note_len = nticks
            self.write_note(note_str, nticks)
            self.put(whitespace)
            return

        # Compute note and release duration.
        note_ticks, rest_ticks = self.get_release(
            coalesce(nticks, self.state.default_note_len)
        )

        if nticks is None and note_ticks == self.state.default_note_len:
            assert rest_ticks == 0
            self.put(note_str + whitespace)
        elif note_ticks + rest_ticks > 0:
            if note_ticks:
                self.write_note(note_str, note_ticks)
            if rest_ticks:
                self.write_note('r', rest_ticks)
            self.put(whitespace)

    def get_release(self, nticks):
        note_ticks = self.state.note_len_calc(nticks)
        name = self.state.note_len_calc.__name__
        if note_ticks > nticks:
            # Redundant staccatos should be filtered out by staccato().
            raise MMKError(
                f'Note length {name}: overlong '
                f'{note_ticks}-ticks from {nticks}-tick note')

        if nticks > 0 and note_ticks <= 0:
            raise MMKError(
                f'Note length {name}: missing '
                f'{note_ticks}-ticks from {nticks}-tick note')

        rest_ticks = nticks - note_ticks
        return note_ticks, rest_ticks

    def write_note(self, note_str: str, nticks: int):
        time_str: str = self._format_time(nticks)
        self.put(f'{note_str}{time_str}')

    NOTES_ONLY = frozenset('abcdefg')

    def parse_notelen(self, char: str, note_len: NoteLenFactory):
        """ Release the next note early.
        If two tildes, release all future notes early.

        Single-note form cannot be followed by lxx.
        TODO: Should ties be allowed?
        """

        def read_release():
            dur, _ = self.stream.get_time()
            self.state.note_len_calc = note_len(dur)

        assert self.stream.get_char() == char
        if self.stream.peek() == char:
            self.stream.get_char()
            # Continue until cancelled.
            read_release()

        else:
            # Release the next note.
            old_state = copy.copy(self.state)
            read_release()
            self.parse_note(allowed_notes=self.NOTES_ONLY)
            self.state = old_state

    @staticmethod
    def _format_time(ntick: Optional[int]) -> str:
        """ Convert a tick duration to a MML "c4" or "c=48"-style duration. """
        if ntick is None:
            return ''

        # If possible, convert to fraction of a measure (c4).
        measure_frac = Fraction(ntick, TICKS_PER_MEASURE)
        if measure_frac.numerator == 1:
            return str(measure_frac.denominator)

        # Otherwise return a tick duration (c=48).
        return f'={ntick}'

    # **** Transpose ****

    def parse_transpose(self) -> None:
        transpose_str, whitespace = self.stream.get_phrase(1)
        transpose = int(transpose_str)

        if transpose not in range(-0x80, 0x80):
            raise MMKError('invalid transpose {}'.format(transpose_str))

        transpose_hex = to_hex(transpose & 0xff)
        self.put('$FA $02 {}'.format(transpose_hex))
        self.put(whitespace)

    # **** volume ****

    def calc_vol(self, in_vol):
        vol = parse_frac(in_vol)
        vol *= self.state.vmod

        if self.state.isvol:
            vol *= 2
        return str(round(vol))

    def parse_vol(self):
        self.stream.skip_chars(1, self.put)
        orig_vol = self.stream.get_int()

        self.state.v = self.calc_vol(orig_vol)
        self.put(self.state.v)

    def parse_vol_hex(self, arg):
        # This both returns the volume and modifies state.
        # Time to throw away state?
        assert self.state is not self.orig_state

        new_vol = self.state.v = self.calc_vol(arg)    # type: str
        hex_vol = to_hex(new_vol)
        return hex_vol

    def parse_vbend(self):
        # Takes a fraction of a quarter note as input.
        # Converts to ticks.
        time, _ = self.stream.get_time()
        vol, whitespace = self.stream.get_phrase(1)

        time_hex = to_hex(time)
        vol_hex = self.parse_vol_hex(vol)

        self.put('$E8 {} {}{}'.format(time_hex, vol_hex, whitespace))

    # **** pan ****

    def calc_pan(self, orig_pan):
        # Convert panning
        if self.state.ispan:
            zeroed_pan = parse_frac(orig_pan) - 64
            scaled_pan = zeroed_pan * self.state.panscale
            return str(round(scaled_pan + 10))
        else:
            return str(orig_pan)

    def parse_pan(self):
        self.stream.skip_chars(1, self.put)
        orig_pan = self.stream.get_int()

        self.state.y = self.calc_pan(orig_pan)
        # Pass the command through.
        self.put(self.state.y)

    def parse_ybend(self):
        duration, _ = self.stream.get_time()
        pan, whitespace = self.stream.get_phrase(1)

        duration_hex = to_hex(duration)
        self.state.y = self.calc_pan(pan)
        pan_hex = to_hex(self.state.y)

        self.put('$DC {} {}{}'.format(duration_hex, pan_hex, whitespace))

    # **** meh ****

    def parse_comment(self, put=True):
        comment = self.stream.get_until(any_of('\n'), strict=False)
        if put:
            self.put(comment)

    def skip_comment(self):
        self.parse_comment(put=False)

    # Multi-word parsing

    def parse_pbend(self):
        # Takes a fraction of a quarter note as input.
        # Converts to ticks.
        delay, _ = self.stream.get_time()
        time, _ = self.stream.get_time()
        note, whitespace = self.stream.get_phrase(1)

        delay_hex = to_hex(delay)
        time_hex = to_hex(time)

        self.put('$DD {} {} {}{}'.format(delay_hex, time_hex, note, whitespace))

    # **** oscillatory effects ****

    def parse_vib(self):
        delay, _ = self.stream.get_time()
        frequency, amplitude, whitespace = self.stream.get_phrase(2)

        delay_hex = to_hex(delay)
        freq_hex = to_hex(parse_frac(frequency))

        self.put('$DE {} {} {}{}'.format(delay_hex, freq_hex, amplitude, whitespace))

    def parse_trem(self):
        delay, _ = self.stream.get_time()
        frequency, amplitude, whitespace = self.stream.get_phrase(2)

        delay_hex = to_hex(delay)
        freq_hex = to_hex(parse_frac(frequency))

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
        curve, rate, whitespace = self.stream.get_phrase(2)

        if instr:
            prefix = '$00 $00'
        else:
            prefix = '$FA $01'

        raw_rate = rate
        rate = parse_hex_only(rate)
        for *curves, begin, max_rate in self._GAINS:
            if curve in curves:
                rate = self._index_check(curve, rate, max_rate)
                self.put('%s %s%s' % (prefix, to_hex(begin + rate), whitespace))
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
        attack, decay, sustain, release, whitespace = self.stream.get_phrase(4)
        if sustain.startswith('full'):
            sustain = '7'

        attack = parse_hex_only(attack)
        decay = parse_hex_only(decay)
        sustain = parse_hex_only(sustain)
        release = parse_hex_only(release)

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
        self.put(fmt.format(to_hex(a), to_hex(b)))
        self.put(whitespace)

    def parse_exp(self, instr: bool):
        release, whitespace = self.stream.get_word()
        with self.set_input(f'-1,-1,full,' + release + whitespace):
            self.parse_adsr(instr)

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
        expr = self.stream.get_until(any_of(')'), strict=True)
        args = [word.strip() for word in expr.split(',')]

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

    def parse_instr(self):
        """ Parse an instrument definition. Define a name for the instrument number.
        Do not place %tune before %instr, it breaks named instruments.

        "foo.brr"
        - %foo=@30

        Define a custom alias.
        bar="foo.brr"
        - %bar=@31
        """
        with self.capture() as fout, self.until_comment():
            input = self.stream.in_str
            self.parse_instruments()
            output = fout.getvalue()

        # Process custom aliases
        if '=' in input and input.index('=') < input.index('"'):
            before_assignment = input.split('=')[0].strip()
            instr_name = before_assignment.split()[-1]
            if not instr_name:
                raise MMKError('invalid alias definition, what is this? ' + input)
        else:
            instr_name = None

        self.put(output)

        if instr_name is None:
            stream = Stream(output, self.defines)
            instr_path, whitespace = stream.get_quoted()
            instr_path = Path(instr_path)
            if instr_path.suffix != '.brr':
                raise MMKError(f'Invalid instrument sample {instr_path} not .brr file')
            instr_name = instr_path.stem

        self.defines[instr_name] = f'@{self.instr_num}'

        self.instr_num += 1

    def parse_tune(self):
        self.smp_num += 1
        # "test.brr" $ad $sr $gain $tune $tune

        self.stream.get_until(any_of('"'), strict=True)
        brr, whitespace = self.stream.get_quoted()

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
        self.smp_num += self.stream.get_int()

    def parse_wave_group(self, is_instruments: bool):
        """
        #samples {
        %wave_group "name" [ntick_playback] [silent|...]

        #instruments {
        %wave_group "0" %adsr -1,-1,-1,0
        """
        name, whitespace = self.stream.get_quoted()
        if name.endswith(".brr"):
            raise MMKError(f'Try removing ".brr" from %wave_group {name}')
        ntick_playback = None

        if not is_instruments:
            ntick_playback = self.stream.get_int(maybe=True)     # Only load the first N ticks
            if ntick_playback is not None:
                whitespace = self.stream.get_spaces(exclude='\n')

        meta = self.wavetable[name]
        waves = self._get_waves_in_group(name, ntick_playback)

        with self.capture() as output, self.until_comment():
            if is_instruments:
                self.parse_instruments()
                self.put(' ' + meta.tuning_str)
                # *ugh* the instrument's tuning value is basically unused
            else:
                self.put(self.stream.get_line())

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
        comments = self.stream.get_line()
        self.stream.skip_chars(1)  # remove trailing newline

        for wave in waves:
            # eh, missing indentation. who cares.
            self.put(f'"{wave}"{whitespace}{after}{comments}\n')
            comments = ''

        if not is_instruments:  # FIXME
            meta.smp_idx = self.smp_num
            meta.nwave = len(waves)
            # meta.nwave is not always equal to len(meta.pitches),
            # since they may be subsampled differently.
            self.smp_num += len(waves)

    WAVE_GROUP_TEMPLATE = '{}-{:03}.brr'

    def _get_waves_in_group(self, name: str, ntick_playback: Optional[int]) -> List[str]:
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

    # Wave sweeps
    _REG = 0xF6
    def put_load_sample(self, smp_idx: int):
        self.put_hex(self._REG, self._get_wave_reg(), smp_idx)

    def _get_wave_reg(self):
        return 0x10 * self.curr_chan + 0x04

    # Echo and FIR

    def parse_fir(self):
        # params = []
        *params, _whitespace = self.stream.get_phrase(8)
        params = [parse_int_hex(param) for param in params]
            # params.append(self.stream.get_int())
            # _whitespace = self.stream.get_line_spaces()

        self.put('$F5 ')
        self.put_hex(*params)

    # self.state:
    # PAN, VOL, INSTR: str (Remove segments?)
    # PANSCALE: Fraction (5/64)
    # ISVOL, ISPAN: bool

    def parse(self) -> str:
        # For exception debug
        try:
            while not self.stream.is_eof():
                # Yeah, simpler this way. But could hide bugs/inconsistencies.
                self.stream.skip_spaces(self.put)

                if self.stream.is_eof():
                    break
                    # Only whitespace left, means already printed, nothing more to do
                self._begin_pos = self.stream.pos

                amk_key = self.amk_keys.longest_prefix(StringSlice(
                        self.stream.in_str,
                        self.stream.pos
                ))
                if amk_key:
                    self.stream.skip_chars(len(amk_key.key), self.put)
                    continue

                char = self.stream.peek()

                # noinspection PyUnreachableCode
                if False:
                    # Do you realize exactly how many bugs I've created
                    # because I accidentally used `if` instead of `elif`?
                    pass

                # Save AMK keys, to skip parsing them later.
                elif char == '"':
                    self.parse_amk_replace()

                # Parse the default AMK commands.
                elif self.state.is_notelen and char in NOTES_WITH_DURATION:
                    self.parse_note()

                elif self.state.is_notelen and char == RELEASE_CHAR:
                    self.parse_notelen(RELEASE_CHAR, release_early)

                elif self.state.is_notelen and char == STACCATO_CHAR:
                    self.parse_notelen(STACCATO_CHAR, staccato)

                elif char == 'v':
                    self.parse_vol()

                elif char == 'y':
                    self.parse_pan()

                elif char in 'q$':
                    self.stream.skip_chars(3, self.put)

                elif char == ';':
                    self.parse_comment()

                elif char == '#':  # instruments{}
                    self.stream.skip_chars(1, self.put)
                    self.stream.skip_spaces(self.put)

                    ret = False

                    def branch(keyword: str, method: Callable):
                        nonlocal ret
                        if self.stream.peek_equals(keyword):
                            self.stream.skip_until('{', self.put)
                            self.stream.skip_chars(1, self.put)
                            method()
                            ret = True

                    branch('samples', self.parse_samples)
                    branch('instruments', self.parse_instruments)
                    branch('spc', self.parse_spc)

                    if ret:
                        continue
                    if self.stream.peek().isnumeric():
                        chan = self.stream.get_char()
                        self.curr_chan = int(chan)
                        self.put(chan)
                    else:
                        self.put(self.stream.get_line())

                elif char == '(':
                    self.stream.skip_chars(1, self.put)
                    if self.stream.peek() == '!':
                        self.stream.skip_chars(1, self.put)
                        self.parse_callback()

                # Begin custom commands.
                elif char == '%':
                    self.stream.skip_chars(1)

                    # NO ARGUMENTS
                    command_case, whitespace = self.stream.get_word()
                    command = command_case.lower()
                    self._command = command

                    if self.subst_define(command_case, whitespace):
                        continue

                    if command == 'mmk0.1':
                        raise Exception("this shouldn't happen")

                    elif command == 'define':
                        key = self.stream.get_word()[0]
                        value = self.stream.get_line()
                        self.defines[key] = value

                    elif command == 'reset':
                        self.state = copy.copy(self.orig_state)
                        assert self.state is not self.orig_state

                    elif command == 'isvol':
                        self.state.isvol = True

                    elif command == 'ispan':
                        self.state.ispan = True

                    elif command == 'notvol':
                        self.state.isvol = False

                    elif command == 'notpan':
                        self.state.ispan = False

                    elif command == 'notelen':
                        self.parse_toggle_notelen()

                    # N ARGUMENTS

                    elif command == 'save':
                        self.parse_save()

                    elif command == 'restore':
                        self.parse_restore()

                    elif command in ['t', 'transpose']:
                        self.parse_transpose()

                    elif command == 'adsr':
                        self.parse_adsr(instr=False)

                    elif command == 'exp':
                        self.parse_exp(instr=False)

                    elif command == 'gain':
                        self.parse_gain(instr=False)

                    # Wavetable sweep
                    elif command == 'wave_sweep':
                        parse_wave_sweep(self)

                    elif command == 'sweep{':
                        parse_parametric_sweep(self, is_legato=True)

                    elif command in ['note_sweep{', 'ns{']:
                        parse_parametric_sweep(self, is_legato=False,
                                               retrigger_sweep=True)

                    # Echo and FIR
                    elif command == 'fir':
                        self.parse_fir()

                    # Volume scaling
                    elif command == 'vmod':
                        arg, _ = self.stream.get_word()
                        self.state.vmod = parse_frac(arg)

                    # Parameter slides
                    elif command in ['vbend', 'vb']:
                        self.parse_vbend()

                    elif command in ['ybend', 'yb']:
                        self.parse_ybend()

                    elif command in ['pbend', 'pb']:
                        self.parse_pbend()

                    # Vibrato/tremolo
                    elif command == 'vib':
                        self.parse_vib()

                    elif command == 'trem':
                        self.parse_trem()

                    # INVALID COMMAND
                    else:
                        raise MMKError('Invalid command ' + command)
                else:
                    self.stream.skip_chars(1, self.put)
                    self.stream.skip_spaces(self.put)

            return self.out.getvalue().strip() + '\n'
        except Exception:
            # Seek at least 100 characters back
            begin_pos = self._begin_pos
            idx = begin_pos
            for i in range(3):
                idx = self.stream.in_str.rfind('\n', 0, idx)
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
            perr(self.stream.in_str[idx:begin_pos] + '...\n')

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

            while not self.stream.is_eof():
                # pos = self.pos
                self.stream.skip_spaces(self.put, exclude=close)
                self._begin_pos = self.stream.pos
                # assert pos == self.pos
                char = self.stream.peek()

                if char in close:
                    self.stream.skip_chars(1, self.put)  # {}, ""
                    self.stream.skip_spaces(self.put, exclude='\n')
                    return

                if char == ';':
                    self.parse_comment()

                elif char == '%':
                    self.stream.skip_chars(1)

                    command_case, whitespace = self.stream.get_word()
                    command = command_case.lower()
                    self._command = command

                    # **** Parse defines ****
                    if self.subst_define(command_case, whitespace):
                        pass
                    # **** Parse commands ****
                    elif command in mapping:
                        mapping[command](self)
                    else:
                        perr(mapping.keys())
                        raise MMKError('Invalid command ' + command)
                else:
                    self.stream.skip_chars(1, self.put)
                    self.stream.skip_spaces(self.put)
        return _parse


    # noinspection PyArgumentList
    parse_instruments = _brace_parser_factory({
        'instr': lambda self: self.parse_instr(),
        'group': lambda self: self.parse_group(),
        'tune': lambda self: self.parse_tune(),
        'gain': lambda self: self.parse_gain(instr=True),
        'adsr': lambda self: self.parse_adsr(instr=True),
        'exp': lambda self: self.parse_exp(instr=True),
        'wave_group': lambda self: self.parse_wave_group(is_instruments=True),
    })

    # noinspection PyArgumentList
    parse_samples = _brace_parser_factory({
        'smp': lambda self: self.parse_smp(),
        'silent': lambda self: self.parse_silent(),
        'wave_group': lambda self: self.parse_wave_group(is_instruments=False),
    })

    # noinspection PyArgumentList
    parse_spc = _brace_parser_factory({})


#### %wave_sweep

T = TypeVar('T')
Timed = Tuple[int, T]


@dataclass
class SweepEvent:
    sample_idx: Optional[int]
    pitch: Optional[float]

    def __bool__(self):
        return any(x is not None for x in dataclasses.astuple(self))


SweepIter = Iterator[Tuple[int, SweepEvent]]
SweepList = List[Tuple[int, SweepEvent]]


class Sweepable(ABC):
    @abstractmethod
    def ntick(self, midi_pitch: Optional[int]) -> int: ...

    @abstractmethod
    def iter(self, midi_pitch: Optional[int]) -> SweepIter: ...


def sweep_chain(sweeps: List[Sweepable], midi_pitch: Optional[int]) -> SweepIter:
    curr_ntick = 0
    for sweep in sweeps:
        for tick, event in sweep.iter(midi_pitch):
            yield (curr_ntick + tick, event)
        curr_ntick += sweep.ntick(midi_pitch)


class PitchedSweep(Sweepable):
    """ Pitched sweep, with fixed wave/pitch rate. """

    def __init__(self, meta: WavetableMetadata):
        self.meta = meta
        self._ntick = meta.ntick

    def ntick(self, midi_pitch: Optional[int]) -> int:
        return self._ntick

    def iter(self, midi_pitch: Optional[int]) -> SweepIter:
        """ Pitched sweep, plays at fixed pitch and rate. midi_pitch is ignored. """
        meta = self.meta

        def tick_range(skip):
            return peekable(itertools.chain(
                range(0, meta.ntick, skip),
                [math.inf],
            ))

        wave_ticks = tick_range(meta.wave_sub)
        pitch_ticks = tick_range(meta.env_sub)

        tick = 0

        while tick < meta.ntick:
            event = SweepEvent(None, None)

            # Wave envelope
            if tick == wave_ticks.peek():
                event.sample_idx = tick // meta.wave_sub
                next(wave_ticks)

            if tick == pitch_ticks.peek():
                env_idx = tick // meta.env_sub
                event.pitch = meta.pitches[env_idx]
                next(pitch_ticks)

            yield (tick, event)

            tick = min(wave_ticks.peek(), pitch_ticks.peek())


# @dataclass
class INote:
    ntick: int


@dataclass
class Note(INote):
    """ A note used in %wave_sweep and %sweep{.
    If `midi_pitch` is set, it overrides the sweep's pitch. """
    ntick: int
    midi_pitch: Optional[int]


@dataclass(frozen=True)
class _ToggleLegato(INote):
    ntick: int = 0

ToggleLegato = _ToggleLegato()
del _ToggleLegato


NoteIter = Iterator[Tuple[int, INote]]


def note_chain(notes: List[INote]) -> NoteIter:
    tick = 0
    for note in notes:
        yield (tick, note)
        tick += note.ntick


DETUNE = 0xEE
LEGATO = '$F4 $01  '
def parse_wave_sweep(self: MMKParser):
    """ Print a wavetable sweep at a fixed rate. """
    name, _ = self.stream.get_quoted()
    note_ntick = self.stream.get_int()     # The sweep lasts for N ticks

    meta = self.wavetable[name]

    sweeps = [PitchedSweep(meta)]
    notes = [ToggleLegato, Note(note_ntick, None)]

    _put_sweep(self, sweeps, notes, meta, is_legato=True)


@dataclass
class SweepState:
    is_legato: bool
    is_detuned: bool


def _put_sweep(
        self: MMKParser,
        sweeps: List[Sweepable],
        notes: List[INote],
        meta: WavetableMetadata,
        is_legato: bool,
        retrigger_sweep: bool = False
):
    """ Write a wavetable sweep. Duration is determined by `notes`.
    If notes[].midi_pitch exists, overrides sweeps[].pitch.

    Used by %wave_sweep and %sweep{.

    # Each note follows a pitch/wave event. It is printed (with the proper
    # begin/end ticks) when the next pitch/wave event begins.

    Workflow: If a note lasts from t0 to t1, the following occurs:
    - end_note(t0)
    - SweepEvent assigns sweep_pitch[t0]
    - and/or Note assigns note_pitch[t0]
    - end_note(t1) writes a note from t0 to t1. midi_pitch() == end of t0.

    TODO:
    - Add datatype for rests
    - Add support for arbitrary events (volume, pan)
    - Add support for retriggering wave envelope
    """
    if getattr(meta, 'nwave', None) is None:
        raise MMKError(f'Did you forget to add #samples{{ %wave_group "{meta.name}" ?')

    # Enable ADSR fade-in
    self.parse_str('%adsr -3,-1,full,0  ')

    # Load silent instrument with proper tuning
    if self.silent_idx is None:
        raise MMKError('cannot %wave_sweep without silent sample defined')
    # @30 to zero out fine-tuning
    self.put(self.defines['silent'])
    # Set coarse tuning
    self.put_hex(0xf3, self.silent_idx, meta.tuning)

    state = SweepState(is_legato, is_detuned=False)
    del is_legato

    # Enable legato
    if state.is_legato:
        self.put('  ')
        self.put(LEGATO)   # Legato glues right+2, and unglues left+right.

    if retrigger_sweep:
        # Sweep once per actual note.
        # Note: rests should not retrigger sweep, only continue or stop sweep.

        for note_and_trailing in split_before(notes, _is_note_trigger):
            _put_single_sweep(self, state, meta, sweeps, note_and_trailing)
    else:
        # Sweep continuously across all notes.
        _put_single_sweep(self, state, meta, sweeps, notes)

    # Cleanup: disable legato and detune.
    if state.is_legato:
        self.put(LEGATO)   # Legato deactivates immediately.
    if state.is_detuned:
        self.put_hex(DETUNE, 0)


def _is_note_trigger(e: INote):
    return isinstance(e, Note)


def _get_pitch(notes: List[INote]) -> Optional[int]:
    for note in notes:
        if isinstance(note, Note):
            return note.midi_pitch
    return None


def _put_single_sweep(
        self: MMKParser,
        state: SweepState,
        meta: WavetableMetadata,
        sweeps: List[Sweepable],
        notes: List[INote],
):
    """ Note: If retriggering is enabled, each note will call this function
    with the same `sweep_list`, but different chunks of `notes`.
    So precompute `sweep_list` for a (dubious) efficiency boost.
    """

    midi_pitch = _get_pitch(notes)

    # Generate iterator of all SweepEvents.
    sweep_iter: SweepIter = sweep_chain(sweeps, midi_pitch)
    sweep_ntick = sum(sweep.ntick(midi_pitch) for sweep in sweeps)

    # Generate iterator of all Notes
    note_iter = note_chain(notes)
    note_ntick = sum(note.ntick for note in notes)

    # Overall event iterator.
    time_event_iter = heapq.merge(sweep_iter, note_iter, key=lambda tup: tup[0])

    #### Write notes.

    note_begin = 0
    def end_note(note_end):
        nonlocal note_begin
        dtime = note_end - note_begin

        if dtime > 0:
            # AddmusicK will glitch out if you write a 1-tick note,
            # followed by instrument/volume changes.

            # TODO unit test
            # If we write a 1-tick terminating note, write a rest instead.
            if note_end == note_ntick and dtime == 1:
                note_str = 'r'
            else:
                note_str = note_name()

            self.put(f'{note_str}={dtime}  ')
            note_begin = note_end

    # Pitch tracking
    note_pitch: int = None
    is_new_note: bool = False
    sweep_pitch: int = None

    def note_name() -> str:
        """ Return note, tie, or pitch from sweep. """
        nonlocal is_new_note
        if note_pitch is not None:
            if is_new_note:
                is_new_note = False
                return format_note(note_pitch)
            else:
                return '^'
        elif sweep_pitch is not None:
            return format_note(sweep_pitch)
        else:
            raise ValueError('_put_sweep missing both note_pitch and sweep_pitch')

    for time, event in time_event_iter:  # type: int, Union[SweepEvent, INote]
        if time >= note_ntick:
            break
        end_note(time)

        if isinstance(event, SweepEvent):
            # Wave envelope
            if event.sample_idx is not None:
                if not 0 <= event.sample_idx < meta.nwave:
                    raise MMKError(
                        f'Cannot sweep to sample {event.sample_idx}, len={meta.nwave}'
                        f' ({meta.name})'
                    )
                self.put_load_sample(meta.smp_idx + event.sample_idx)
            # Pitch envelope
            if event.pitch is not None:
                # Decompose sweep pitch into integer and detune.
                sweep_pitch = int(event.pitch)
                detune = event.pitch - sweep_pitch

                detune_int = int(detune * 256)
                state.is_detuned = (detune_int != 0)

                # Write detune value immediately (begins at following note).
                self.put_hex(DETUNE, detune_int)

        elif isinstance(event, Note):
            note_pitch = event.midi_pitch
            is_new_note = True

        elif event is ToggleLegato:
            state.is_legato = not state.is_legato
            self.put('  ' + LEGATO)

        else:
            raise TypeError(f'invalid sweep event type={type(event)}, programmer error')

    if meta.silent and sweep_ntick < note_ntick:
        # Add GAIN fadeout.
        end_note(sweep_ntick)
        # GAIN starts when the following note starts.
        self.parse_str('%gain down $18  ')

    # End final note.
    end_note(note_ntick)


### %sweep{

class LinearSweep(Sweepable):
    def __init__(self, sweep: range, ntick: int, pitch_scaling: float, root_pitch: int):
        self.sweep = sweep  # Range of sweep
        self.nsweep = len(sweep)

        self._ntick_unscaled = ntick
        self.pitch_scaling = pitch_scaling
        self.root_pitch = root_pitch

    def ntick(self, midi_pitch: Optional[int]) -> int:
        """ ntick /= (f/f0) ** scaling """
        if midi_pitch is None:
            return self._ntick_unscaled

        dpitch = (midi_pitch - self.root_pitch)
        freq_ratio = 2 ** (dpitch / 12)
        return round(self._ntick_unscaled / (freq_ratio ** self.pitch_scaling))

    def iter(self, midi_pitch: Optional[int]) -> SweepIter:
        """ Unpitched linear sweep, with fixed endpoints and duration.
        Created using the `[a,b) time` notation.
        """
        prev_tick = -1
        ntick = self.ntick(midi_pitch)

        for sweep_idx in range(self.nsweep):
            tick = ceildiv(sweep_idx * ntick, self.nsweep)
            if tick > prev_tick:
                event = SweepEvent(self.sweep[sweep_idx], None)
                yield (tick, event)


ONLY_WHITESPACE = ' \t\n\r\x0b\f'

def parse_parametric_sweep(self: MMKParser, is_legato: bool,
                           retrigger_sweep: bool = False):
    """ Read parameters, and print a sweep with fixed duration.

    %sweep{ "name"
        [begin,end) beats   # Increasing intervals have step 1.
        [,end] beats        # Decreasing intervals have step -1.
        [begin,end) beats ~scaling
            # Notes above/below av have speed multiplied by (f/f0) ** scaling.

        TODO:
        =       # `env_sub` ticks per wave.
        =/3     # `env_sub*3` ticks per wave.
        =/3 ~scaling    # Apply scaling to above.
        :

        # Duration in beats, separate from outside lxx events.
        l/4
        o4 c1 c/2 c c

        # Loops are unrolled.
        [c <b >]5
    }
    """
    stream = self.stream
    stream.skip_spaces()

    # Get name
    name, _ = stream.get_quoted()
    meta = self.wavetable[name]

    # Get sweep, duration pairs
    sweeps = []
    stream.skip_spaces()
    while stream.peek() != ':':
        # TODO unit test comments in header, especially skipping spaces.
        if stream.peek() == ';':
            self.skip_comment()
            stream.skip_spaces()
            continue

        sweep_str, _ = stream.get_word(ONLY_WHITESPACE)
        if sweep_str == '=':
            raise MMKError('sweep{ = at fixed rate is not supported yet')

        # [x,y)
        sweep_range = parse_wave_range(sweep_str, meta.nwave)

        # Read sweep duration
        if stream.peek() == '>':
            # Rate: Fraction of all waves, per tick
            stream.get_char()
            word, _ = stream.get_word()

            if '/' in word or '.' in word:
                nwave_percent_per_tick = parse_frac(word)
            else:
                nwave_percent_per_tick = int(word)
            ntick = round(1 / nwave_percent_per_tick * len(sweep_range) / meta.nwave)
        elif stream.peek() == '*':
            # One wave every X ticks
            stream.get_char()
            duration_mul = parse_frac(stream.get_word()[0])
            ntick = round(meta.nwave * duration_mul)
        else:
            ntick, _ = stream.get_time()

        if ntick is None:
            raise MMKError('failed to specify sweep time')


        # Read speed scaling exponent.
        if stream.peek() == '~':
            stream.skip_chars(1)
            pitch_scaling = safe_eval(stream.get_word()[0], numbers.Real)
        else:
            pitch_scaling = 0

        sweeps.append(LinearSweep(
            sweep_range,
            ntick,
            pitch_scaling,
            meta.root_pitch,
        ))
        stream.skip_spaces()
        # stream.skip_spaces(exclude=set(WHITESPACE) - set(ONLY_WHITESPACE))

    # I can't remember why I ever marked colons as whitespace...
    # It's not used in standard AMK MML.
    # Using colon as a syntactic separator is creating a world of pain.
    _separator = stream.get_char()
    stream.skip_spaces()

    # Get notes
    notes = []
    note_chars = set('abcdefg')

    octave = None
    default_ntick = None

    while stream.peek() != '}':
        c = stream.get_char()

        # noinspection PyUnreachableCode
        if False: pass

        elif c == ';':
            self.skip_comment()

        # octave
        elif c == 'o':
            octave = int(stream.get_char())
        elif c == '>':
            octave += 1
        elif c == '<':
            octave -= 1

        # Legato/slur toggle
        elif c == '_':
            notes.append(ToggleLegato)

        # note length
        elif c == 'l':
            default_ntick, _ = stream.get_time()
        # notes
        elif c in note_chars:

            # Note pitch
            # TODO note to midi function?

            sharp_flat = stream.peek()
            if sharp_flat in accidental2pitch:
                stream.skip_chars(1)
                dpitch = accidental2pitch[sharp_flat]
            else:
                dpitch = 0
            if octave is None:
                raise MMKError('You must assign octave within sweep{}')
            midi_pitch = note2pitch[c] + dpitch + OCTAVE * (octave + 1)

            # Note duration
            ntick, _ = stream.get_time()
            try:
                ntick = coalesce(ntick, default_ntick)
            except TypeError:
                raise MMKError(
                    'You must assign lxx within sweep{} before entering untimed notes')

            notes.append(Note(ntick, midi_pitch))
        # ties
        elif c == '^':
            tie_ntick, _ = stream.get_time()
            notes[-1].ntick += tie_ntick

        stream.skip_spaces()

    # Eat close }
    stream.skip_chars(1)

    _put_sweep(self, sweeps, notes, meta, is_legato, retrigger_sweep)


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
