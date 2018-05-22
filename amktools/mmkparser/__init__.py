#!/usr/bin/env python3

# MMK Parser for AddMusicK
# Written by nobody1089
# Released under the WTFPL

import argparse
import os
from ruamel.yaml import YAML
from fractions import Fraction

# Pretend macros don't exist, because they're too much trouble.
# Commands are parsed at macro-define, not macro-eval.
# Additionally, we cannot identify instrument macros. Doesn't matter if I remove segmenting.
# The only way to fix that would be to expand macros, which would both complicate the program and
# make the generated source less human-readable.

# Now I am keeping the segmentation concept, but discarding the volume preservation. Too difficult.
from pathlib import Path
from typing import Dict, List, Union


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


def frac2hex(in_frac):
    return '$' + hex(int(in_frac))[2:].zfill(2)


QUARTER_TO_TICKS = 0x30


def vol_midi2smw(midi_vol):
    midi_vol = parse_frac(midi_vol)
    fractional = midi_vol / 127
    smw_vol = fractional * 255
    return round(smw_vol)


DELIMITERS = ' \t\n\r\x0b\f:,"'


# Focus on parsing text.
class MMKParser:
    SHEBANG = '%mmk0.1'
    SEGMENT = str

    def __init__(self, in_str, tuning: Union[dict, None]):
        self.in_str = in_str
        self.tuning = tuning

        self.orig_state = {
            'isvol': False, 'ispan': False, 'panscale': Fraction('5/64'), 'vmod': Fraction(1)}
        self.state = self.orig_state.copy()
        self.defines = {}  # type: Dict[str, str]

        self.currseg = -1
        self.seg_text = {}  # type: Dict[int, List[self.SEGMENT]]
        self.pos = 0
        self.out = []

        self.is_quote = False

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

    # **** Parsing ****
    def skip_chars(self, num, keep: bool = True) -> None:
        """ Skips the specified number of characters.
        :param num: Number of characters to skip.
        :param keep: Whether to write characters.
        :return: None
        """

        pos = self.pos
        skipped = self.in_str[pos: pos + num]
        self.pos = min(pos + num, self.size())
        if keep:
            self.put(skipped)

    # ****

    def get_word(self):
        """ Removes all leading spaces, but only trailing spaces up to the first \n.
        That helps preserve formatting.
        :return: (word, trailing whitespace)
        """

        self.skip_spaces(False, exclude='')

        word = ''
        while not self.peek() in DELIMITERS:
            word += self.get_char()
        trailing = self.skip_spaces(False, exclude='\n')

        if word.startswith('%'):
            word = self.defines.get(word[1:], word)
        return word, trailing

    def get_line(self):
        self.skip_spaces(False, exclude='')

        line = ''
        while not self.peek() == '\n':  # fixme shlemiel the painter
            line += self.get_char()
        trailing = ''
        return line, trailing

    def get_int(self):
        # Gets an integer. No whitespace needed.
        buffer = ''
        while self.peek().isdigit():
            buffer += self.get_char()
        return parse_int_round(buffer)

    def skip_spaces(self, keep, exclude=''):
        # Optional exclude newlines, for example. Useful for preserving formatting.
        skipped = ''

        delimiters = set(DELIMITERS) - set(exclude)
        while not self.is_eof() and self.peek() in delimiters:
            if keep:
                self.put(self.peek())
            skipped += self.get_char()

        return skipped

    def put(self, pstr):
        if self.currseg == -1:
            self.out.append(pstr)
        elif self.currseg in self.seg_text:
            self.seg_text[self.currseg].append(pstr)
        else:
            self.seg_text[self.currseg] = [pstr]

    # **** BEGIN CALCULATION FUNCTIONS ****
    def calc_vol(self, in_vol):
        vol = parse_frac(in_vol)
        vol *= self.state['vmod']

        if self.state['isvol']:
            vol *= 2
        return str(round(vol))

    # Begin parsing functions!
    def parse_vol(self):
        self.skip_chars(1, keep=False)
        self.skip_spaces(True)
        orig_vol = self.get_int()

        self.state['vol'] = self.calc_vol(orig_vol)
        self.put('v' + self.state['vol'])

    def parse_vol_hex(self, arg):
        # This both returns the volume and modifies state.
        # Time to throw away state?
        new_vol = self.state['vol'] = self.calc_vol(arg)
        hex_vol = hex(int(new_vol))[2:].zfill(2)
        return '$' + hex_vol + ' '

    def parse_pan(self):
        self.skip_chars(1, keep=False)
        self.skip_spaces(True)
        orig_pan = self.get_int()

        # Convert panning
        if self.state['ispan']:
            zeroed_pan = parse_frac(orig_pan) - 64
            scaled_pan = zeroed_pan * self.state['panscale']
            self.state['pan'] = str(round(scaled_pan + 10))
        else:
            self.state['pan'] = str(orig_pan)
        # Pass the command through.
        self.put('y' + self.state['pan'])

    def skip_until(self, end):
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

    def parse_braces(self):
        self.skip_until('}')

    def parse_quotes(self):
        # TODO what is this?
        if self.is_quote is True:
            self.skip_chars(1, keep=True)
            self.is_quote = False
        else:
            final_char = self.skip_until('="')
            # End quote immediately = no longer quote.
            # Other delimiters = continue parsing remaining text.
            if final_char != '"':
                self.is_quote = True

    # Multi-word parsing

    def parse_tune(self, brr, adsr, whitespace):
        if self.tuning is None:
            print('Cannot use %tune without a tuning file')
            raise ValueError

        tuning = self.tuning[brr]
        self.put('{} {} {}{}'.format(brr, tuning, adsr, whitespace))

    def parse_pbend(self, delay, time, note, whitespace):
        # Takes a fraction of a quarter note as input.
        # Converts to ticks.
        delay_hex = frac2hex(parse_frac(delay) * QUARTER_TO_TICKS)
        time_hex = frac2hex(parse_frac(time) * QUARTER_TO_TICKS)

        self.put('$DD {} {} {}{}'.format(delay_hex, time_hex, note, whitespace))

    def parse_vbend(self, time, vol, whitespace):
        # Takes a fraction of a quarter note as input.
        # Converts to ticks.
        time_hex = frac2hex(parse_frac(time) * QUARTER_TO_TICKS)
        vol_hex = self.parse_vol_hex(vol)

        self.put('$E8 {} {}{}'.format(time_hex, vol_hex, whitespace))

    def parse_vib(self, delay, frequency, amplitude, whitespace):
        delay_hex = frac2hex(parse_frac(delay) * QUARTER_TO_TICKS)
        freq_hex = frac2hex(parse_frac(frequency))

        self.put('$DE {} {} {}{}'.format(delay_hex, freq_hex, amplitude, whitespace))

    def parse_trem(self, delay, frequency, amplitude, whitespace):
        delay_hex = frac2hex(parse_frac(delay) * QUARTER_TO_TICKS)
        freq_hex = frac2hex(parse_frac(frequency))

        self.put('$E5 {} {} {}{}'.format(delay_hex, freq_hex, amplitude, whitespace))

    _GAINS = [
        # curve, begin, max_rate
        ['direct', 0x00],
        ['down', 0x80],
        ['exp', 0xa0],
        ['up', 0xc0],
        ['bent', 0xe0],
        [None, 0x100],
    ]

    for i in range(len(_GAINS) - 1):
        _GAINS[i].append(_GAINS[i + 1][1] - _GAINS[i][1])
    _GAINS = _GAINS[:-1]

    def parse_gain(self, curve, rate, whitespace):
        # Look for a matching GAIN value, ensure the input rate lies in-bounds,
        # then write a hex command.
        raw_rate = rate
        rate = parse_int_hex(rate)
        for curve_, begin, max_rate in self._GAINS:
            if curve_ == curve:
                if rate not in range(max_rate):
                    print('Invalid rate %s for curve %s (rate < %s)' %
                          (raw_rate, curve, hex(max_rate)))
                    raise ValueError

                self.put('$ED $80 %s%s' % (frac2hex(begin + rate), whitespace))
                return

        print('Invalid gain %s, options are:' % repr(curve))
        for curve, _, max_rate in self._GAINS:
            print('%s (rate < %s)' % (curve, hex(max_rate)))
        raise ValueError

    # self.state:
    # PAN, VOL, INSTR: str (Remove segments?)
    # PANSCALE: Fraction (5/64)
    # ISVOL, ISPAN: bool

    def parse(self):
        # For exception debug
        command = None
        begin_pos = 0
        try:
            # Remove the header. TODO increment pos instead.
            if self.in_str.startswith(self.SHEBANG):
                self.in_str = self.in_str[len(self.SHEBANG):].lstrip()
            else:
                # print('Missing header "%mmk0.1"')
                # print('Ignoring error...')
                pass

            while self.pos < self.size():
                self.skip_spaces(
                    True)  # Yeah, simpler this way. But could hide bugs/inconsistencies.
                if self.pos == self.size():
                    break
                    # Only whitespace left, means already printed, nothing more to do
                begin_pos = self.pos
                char = self.peek()

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

                if char == '{':
                    self.parse_braces()
                    continue
                #
                # if char == '"':
                #     self.skip_until('"')
                #     continue

                # Begin custom commands.
                if char == '%':
                    self.skip_chars(1, keep=False)

                    # NO ARGUMENTS
                    command_case, trailing = self.get_word()
                    command = command_case.lower()

                    if command_case in self.defines:
                        self.put(self.defines[command_case] + trailing)
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

                    if command == 'tune':
                        brr = self.get_quotes()
                        adsr = self.get_line()
                        self.parse_tune(brr, adsr, '')

                    # ONE ARGUMENT
                    arg, trailing = self.get_word()

                    if command == 'vmod':
                        self.state['vmod'] = parse_frac(arg)
                        continue

                    # 2 ARGUMENTS
                    arg2, trailing = self.get_word()
                    if command == 'vbend':
                        self.parse_vbend(arg, arg2, trailing)
                        continue

                    if command == 'gain':
                        self.parse_gain(arg, arg2, trailing)
                        continue

                    # 3 ARGUMENTS
                    arg3, trailing = self.get_word()

                    if command == 'vib':
                        self.parse_vib(arg, arg2, arg3, trailing)
                        continue

                    if command == 'trem':
                        self.parse_trem(arg, arg2, arg3, trailing)
                        continue

                    if command == 'pbend':
                        self.parse_pbend(arg, arg2, arg3, trailing)
                        continue

                    # 4 ARGUMENTS
                    arg4, trailing = self.get_word()
                    if command == 'adsr':
                        print('ADSR not implemented...')
                        continue  # TODO: ADSR

                    # INVALID COMMAND
                    raise ValueError('Invalid command ' + command)
                else:
                    self.skip_chars(1, keep=True)
                    self.skip_spaces(True)
                    # done processing the command

            return ''.join(self.out).strip() + '\n'
        except Exception:
            print()
            print('Last command "%' + command + '" Context:')
            print('...' + self.in_str[begin_pos - 100: begin_pos] + '...\n')
            raise


def remove_ext(path):
    head = os.path.splitext(path)[0]
    return head


TUNING = 'tuning.yaml'
SUFFIX = '.out.txt'

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
        description='Parse one or more MMK files to a single AddmusicK source file.',
        epilog='''Examples:
`mmk_parser infile.txt`                 outputs to infile.out.txt
`mmk_parser infile.txt infile2.txt`     outputs to infile.out.txt
`mmk_parser infile.txt -o outfile.txt`  outputs to outfile.txt''')

    parser.add_argument('files', help='Input files, will be concatenated', nargs='+')
    parser.add_argument('-t', '--tuning',
                        help='Tuning file produced by convert_brr (defaults to {})'.format(TUNING))
    parser.add_argument('-o', '--outpath', help='Output path (if omitted)')
    args = parser.parse_args()

    # FILES
    inpaths = args.files
    first_path = inpaths[0]

    datas = []
    for inpath in inpaths:
        with open(inpath) as ifile:
            datas.append(ifile.read())
    datas.append('\n')
    in_str = '\n'.join(datas)

    # TUNING
    if 'tuning' in args:
        tuning_path = args.tuning
    else:
        tuning_path = str(Path(first_path, '..', TUNING).resolve())
    try:
        with open(tuning_path) as f:
            tuning = yaml.load(f)
            if type(tuning) != dict:
                raise ValueError('invalid tuning file {}, must be YAML key-value map'.format(tuning_path))
    except FileNotFoundError:
        tuning = None

    # OUT PATH
    if 'outpath' in args:
        outpath = args.outpath
    else:
        outpath = remove_ext(first_path) + SUFFIX

    # parse

    parser = MMKParser(in_str, tuning)
    outstr = parser.parse()
    if outstr is None:
        exit(2)

    with open(outpath, 'w') as ofile:
        ofile.write(outstr)

    # subprocess.call('build.cmd')
