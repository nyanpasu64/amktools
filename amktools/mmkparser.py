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
from pathlib import Path
from typing import Dict, List, Union, Pattern, Tuple

from ruamel.yaml import YAML


# We cannot identify instrument macros.
# The only way to fix that would be to expand macros, which would both complicate the program and
# make the generated source less human-readable.


class MMKError(ValueError):
    pass


def perr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


yaml = YAML(typ='safe')


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
    value = '$%02x' % int(in_frac)
    return value


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


class MMKParser:
    SHEBANG = '%mmk0.1'
    SEGMENT = str

    def __init__(self, in_str: str, tuning: Union[dict, None]):
        self.in_str = in_str
        self.tuning = tuning

        self.orig_state = {
            'isvol': False, 'ispan': False, 'panscale': Fraction('5/64'), 'vmod': Fraction(1)}
        self.state = self.orig_state.copy()
        self.defines = {}  # type: Dict[str, str]

        # unused
        self.currseg = -1
        self.seg_text = {}  # type: Dict[int, List['self.SEGMENT']]

        self.pos = 0
        self.out = []

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

    @contextmanager
    def end_at(self, sub):
        string = self.in_str
        begin = self.pos

        self.in_str = self.get_until(sub, strict=False)

        end = self.pos
        self.pos = 0
        yield

        if begin + self.pos != end:
            raise Exception('Bounded parsing error, parsing ended at {} but region ends at {}'
                            .format(begin + self.pos, end))
        self.in_str = string
        self.pos = end

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
            raise MMKError('Tried to get word where none exists (invalid command or missing arguments?)')
        whitespace = self.get_spaces(exclude='\n')

        if word.startswith('%'):
            word = self.defines.get(word[1:], word)     # dead code?
        return word, whitespace

    def get_words(self, n: int) -> List[str]:
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
        self.skip_spaces(False, exclude='')

        line = ''
        while not self.peek() == '\n':  # fixme shlemiel the painter
            line += self.get_char()
        whitespace = ''
        return line, whitespace

    # Returns parse (doesn't fetch trailing whitespace)
    def get_int(self):
        buffer = ''
        while self.peek().isdigit():
            buffer += self.get_char()
        return parse_int_round(buffer)

    def put(self, pstr):
        if self.currseg == -1:
            self.out.append(pstr)
        elif self.currseg in self.seg_text:
            self.seg_text[self.currseg].append(pstr)
        else:
            self.seg_text[self.currseg] = [pstr]

    # Begin parsing functions!
    def parse_define(self, command_case, whitespace):
        """ TODO Parse literal define, passthrough. """
        if command_case in self.defines:
            self.put(self.defines[command_case] + whitespace)
            return True
        return False

    # **** Transpose ****

    def parse_transpose(self) -> None:
        transpose_str, whitespace = self.get_words(1)
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
        self.skip_spaces(True)
        orig_vol = self.get_int()

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
        self.skip_spaces(True)
        orig_pan = self.get_int()

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

    def parse_tune(self):
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

    def parse_gain(self, curve, rate, whitespace, *, instr):
        # Look for a matching GAIN value, ensure the input rate lies in-bounds,
        # then write a hex command.

        if instr:
            prefix = '$00 $00'
        else:
            prefix = '$FA $01'

        raw_rate = rate
        rate = parse_int_hex(rate)
        for *curves, begin, max_rate in self._GAINS:
            if curve in curves:
                if rate not in range(max_rate):
                    perr('Invalid rate %s for curve %s (rate < %s)' %
                         (raw_rate, curve, hex(max_rate)))
                    raise MMKError

                self.put('%s %s%s' % (prefix, int2hex(begin + rate), whitespace))
                return

        perr('Invalid gain %s, options are:' % repr(curve))
        for curve, _, max_rate in self._GAINS:
            perr('%s (rate < %s)' % (curve, hex(max_rate)))
        raise MMKError

    def parse_adsr(self, attack: str, decay: str, sustain: str, release: str, instr: bool):
        """
        Parse ADSR command.
        :param attack: Attack speed (0-15)
        :param decay: Decay speed (0-7)
        :param sustain: Sustain volume (0-7)
        :param release: Release speed (0-31)
        :param instr: Whether ADSR command occurs in instrument definition (or MML command)
        """
        if sustain.startswith('full'):
            sustain = '7'
        # if release.startswith('inf'):
        #     release = '0'

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

    @staticmethod
    def _index_check(caption, val, end):
        if val < 0:
            val += end
        if val not in range(end):
            raise MMKError('Invalid ADSR {} {} (must be < {})'.format(caption, val, end))
        return val


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

                    def branch(keyword, method):
                        if self.in_str.startswith(keyword, self.pos):
                            self.skip_until('{')
                            self.skip_chars(1, keep=True)
                            method()

                    branch('samples', self.parse_instruments)     # FIXME
                    branch('instruments', self.parse_instruments)


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
                        value = self.get_line()[0]
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

                    if command == 'gain':
                        self.parse_gain(arg, arg2, whitespace, instr=False)
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

                    # 4 ARGUMENTS
                    arg4, whitespace = self.get_word()
                    if command == 'adsr':
                        self.parse_adsr(arg, arg2, arg3, arg4, instr=False)
                        self.put(whitespace)
                        continue

                    # INVALID COMMAND
                    raise MMKError('Invalid command ' + command)
                else:
                    self.skip_chars(1, keep=True)
                    self.skip_spaces(True)

            return ''.join(self.out).strip() + '\n'
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

    def parse_instruments(self, close='}'):
        """
        Parses #instruments{...} blocks. Eats trailing close-brace.
        Also used for parsing quoted BRR filenames within #instruments.
        :param close: Which characters close the current block.
        :return: None
        """
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
                    continue

                # **** Parse commands ****
                if command == 'tune':
                    self.parse_tune()
                    continue

                arg, whitespace = self.get_word()
                arg2, whitespace = self.get_word()
                if command == 'gain':
                    self.parse_gain(arg, arg2, whitespace, instr=True)
                    continue

                arg3, whitespace = self.get_word()
                arg4, whitespace = self.get_word()
                if command == 'adsr':
                    self.parse_adsr(arg, arg2, arg3, arg4, instr=True)
                    self.put(whitespace)
                    continue

                raise MMKError('Invalid command ' + command)
            else:
                self.skip_chars(1, keep=True)
                self.skip_spaces(True)


def remove_ext(path):
    head = os.path.splitext(path)[0]
    return head


from amktools.common import TUNING_PATH
SUFFIX = '.txt'


ERR = 1

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
                        help='Tuning file produced by convert_brr (defaults to {})'.format(TUNING_PATH))
    parser.add_argument('-o', '--outpath', help='Output path (if omitted)')
    args = parser.parse_args(args)

    # FILES
    inpaths = args.files
    first_path = inpaths[0]

    datas = []
    for _inpath in inpaths:
        with open(_inpath) as ifile:
            datas.append(ifile.read())
    datas.append('\n')
    in_str = '\n'.join(datas)

    # TUNING
    if 'tuning' in args:
        tuning_path = args.tuning
    else:
        tuning_path = str(Path(first_path, '..', TUNING_PATH).resolve())
    try:
        with open(tuning_path) as f:
            tuning = yaml.load(f)
        if type(tuning) != dict:
            perr('invalid tuning file {}, must be YAML key-value map'.format(tuning_path))
            return ERR
    except FileNotFoundError:
        tuning = None

    # OUT PATH
    if 'outpath' in args:
        outpath = args.outpath
    else:
        outpath = remove_ext(first_path) + SUFFIX

    for _inpath in inpaths:
        if Path(outpath).resolve() == Path(_inpath).resolve():
            perr('Error: Output file {} will overwrite an input file!'.format(outpath))
            if '.txt' in _inpath.lower():
                perr('Try renaming input files to .mmk')
            return ERR

    # PARSE
    parser = MMKParser(in_str, tuning)
    try:
        outstr = parser.parse()
    except MMKError as e:
        if str(e):
            perr('Error:', str(e))
        return ERR

    with open(outpath, 'w') as ofile:
        ofile.write(outstr)

    return 0


if __name__ == '__main__':
    ret = main(sys.argv[1:])
    exit(ret)
