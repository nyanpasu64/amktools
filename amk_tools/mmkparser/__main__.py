#!/usr/bin/env python3

# MMK Parser for AddMusicK
# Written by nobody1089
# Released under the WTFPL

import argparse
import subprocess

from fractions import Fraction

# Pretend macros don't exist, because they're too much trouble.
# Commands are parsed at macro-define, not macro-eval.
# Additionally, we cannot identify instrument macros. Doesn't matter if I remove segmenting.
# The only way to fix that would be to expand macros, which would both complicate the program and
# make the generated source less human-readable.

# Now I am keeping the segmentation concept, but discarding the volume preservation. Too difficult.


def parse_int(instr):
    return int(parse_frac(instr))


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
    # Divide by 127, square, multiply by 255.
    # MIDI vol is 4rt(real)
    # SMW vol is 2rt(real) = (4rt(real))^2

    # Uh yeah, scratch the squaring!

    midi_vol = parse_frac(midi_vol)
    fractional = midi_vol / 127
    smw_vol = fractional * 255
    return round(smw_vol)


class ExceptionException(Exception):
    pass


DELIMITERS = ' \t\n\r\x0b\f:,'

# Focus on parsing text.
class MMKParser:
    def curr_char(self):
        return self.in_str[self.pos]

    # Begin tokenization/manipulation functions
    def skip_chars(self, num, keep=True):
        # Skips the specified number of characters.
        for x in range(num):
            if self.pos == len(self.in_str):
                break
            if keep:
                self.put(self.curr_char())
            self.pos += 1

    def get_word(self):
        # This removes all leading spaces, but only trailing spaces up to the first \n.
        # That helps preserve formatting.
        self.skip_spaces(False, exclude='')

        buffer = ''
        while not self.curr_char() in DELIMITERS:
            buffer += self.curr_char()
            self.pos += 1
        trailing = self.skip_spaces(False, exclude='\n')

        if buffer[0] == '%':
            buffer = self.defines.get(buffer[1:], buffer)
        return buffer, trailing

    def get_line(self):
        self.skip_spaces(False, exclude='')

        buffer = ''
        while not self.curr_char() == '\n':
            buffer += self.curr_char()
            self.pos += 1
        trailing = ''
        return buffer, trailing


    def get_int(self):
        # Gets an integer. No whitespace needed.
        buffer = ''
        while self.curr_char().isdigit():
            buffer += self.curr_char()
            self.pos += 1
        return parse_int(buffer)

    def skip_spaces(self, keep, exclude=''):
        instring = self.in_str
        # Optional exclude newlines, for example. Useful for preserving formatting.
        if self.pos == len(self.in_str):
            return ''

        after = ''

        while self.curr_char() in DELIMITERS and self.curr_char() not in exclude:
            after += self.curr_char()
            if keep:
                self.put(self.curr_char())
            self.pos += 1
            if self.pos == len(self.in_str):
                break
        return after

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
        #
        #     return str(vol_midi2smw(in_vol))
        # else:
        #     return str(in_vol)
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
        instring = self.in_str
        self.skip_chars(1, keep=True)
        end_pos = instring.find(end, self.pos)
        if end_pos == -1:
            end_pos = self.inlen

        # The delimiter is skipped as well.
        # If end_pos == self.inlen, skip_chars handles the OOB case by not reading the extra char.
        self.skip_chars(end_pos - self.pos + 1, keep=True)

        return self.in_str[end_pos]

    def parse_comment(self):
        self.skip_until('\n')

    def parse_braces(self):
        self.skip_until('}')

    def parse_quotes(self):
        if self.is_quote is True:
            self.skip_chars(1, keep=True)
            self.is_quote = False
        else:
            final_char = self.skip_until('="')
            # End quote immediately = no longer quote.
            # Other delimiters = continue parsing remaining text.
            if final_char != '"':
                self.is_quote = True

    def parse_pbend(self, delay, time, note, after):
        # Takes a fraction of a quarter note as input.
        # Converts to ticks.
        delay_hex = frac2hex(parse_frac(delay) * QUARTER_TO_TICKS)
        time_hex = frac2hex(parse_frac(time) * QUARTER_TO_TICKS)

        self.put('$DD {} {} {}{}'.format(delay_hex, time_hex, note, after))

    def parse_vbend(self, time, vol, after):
        # Takes a fraction of a quarter note as input.
        # Converts to ticks.
        time_hex = frac2hex(parse_frac(time) * QUARTER_TO_TICKS)
        vol_hex = self.parse_vol_hex(vol)

        self.put('$E8 {} {}{}'.format(time_hex, vol_hex, after))

    def parse_vib(self, delay, frequency, amplitude, after):
        delay_hex = frac2hex(parse_frac(delay) * QUARTER_TO_TICKS)
        freq_hex = frac2hex(parse_frac(frequency))

        self.put('$DE {} {} {}{}'.format(delay_hex, freq_hex, amplitude, after))

    def parse_trem(self, delay, frequency, amplitude, after):
        delay_hex = frac2hex(parse_frac(delay) * QUARTER_TO_TICKS)
        freq_hex = frac2hex(parse_frac(frequency))

        self.put('$E5 {} {} {}{}'.format(delay_hex, freq_hex, amplitude, after))

    # PAN, VOL, INSTR: str (Remove segments?)
    # PANSCALE: Fraction (5/64)
    # ISVOL, ISPAN: bool

    def __init__(self, instring):
        self.in_str = instring
        self.orig_state = {'isvol': False, 'ispan': False, 'panscale': parse_frac('5/64'), 'vmod': Fraction(1)}
        self.state = self.orig_state.copy()
        self.seg_states = {}
        self.defines = {}

        self.currseg = -1
        self.seg_text = {}
        self.inlen = len(self.in_str)
        self.pos = 0
        self.out = []

        self.is_quote = False

    def parse(self):
        # For exception debug
        command = None
        begin_pos = 0
        try:
            # Remove the header.
            if self.in_str.startswith('%mmk0.1'):
                self.in_str = self.in_str[len('%mmk0.1'):].lstrip()
                self.inlen = len(self.in_str)
            else:
                print('Missing header "%mmk0.1"')
                print('Ignoring error...')
                # print('Press Enter to exit.')
                # input()
                # return None

            while self.pos < self.inlen:
                self.skip_spaces(True)    # Yeah, simpler this way. But could hide bugs/inconsistencies.
                if self.pos == self.inlen:
                    break
                    # Only whitespace left, means already printed, nothing more to do
                begin_pos = self.pos
                char = self.curr_char()

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
                    commandCase, trailing = self.get_word()
                    command = commandCase.lower()

                    if commandCase in self.defines:
                        self.put(self.defines[commandCase] + trailing)
                        continue

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

                    if command == 'mmk0.1':
                        continue

                    # ONE ARGUMENT
                    arg, trailing = self.get_word()

                    if command == 'vmod':
                        self.state['vmod'] = parse_frac(arg)
                        continue

                    # if command == 'segment':
                    #     self.currseg = arg
                    #     self.seg_states[self.currseg] = self.state.copy()
                    #     continue
                    #
                    # if command == 'miniprint':
                    #     this_seg = ''.join(self.seg_text[arg])
                    #     self.put(this_seg)
                    #     continue
                    #
                    # if command == 'setpanscale':
                    #     self.state['panscale'] = parse_frac(arg)
                    #     continue

                    # 2 ARGUMENTS
                    arg2, trailing = self.get_word()
                    if command == 'vbend':
                        self.parse_vbend(arg, arg2, trailing)
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
                        continue    # TODO: ADSR

                    # INVALID COMMAND
                    raise ExceptionException('Invalid command '+command)
                else:
                    self.skip_chars(1, keep=True)
                    self.skip_spaces(True)
                    # done processing the command

            return ''.join(self.out).strip() + '\n'
        except Exception as ex:
            print('Last command "%' + command + '" Context:')
            print('...' + self.in_str[begin_pos-100 : begin_pos] + '...\n')
            raise ex


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Parse one or more MMK files to a single AddmusicK source file.',
        epilog=
'''Examples:
`mmk_parser infile.txt`                 outputs to infile.out.txt
`mmk_parser infile.txt infile2.txt`     outputs to infile.out.txt
`mmk_parser infile.txt -o outfile.txt`  outputs to outfile.txt
''')
    parser.add_argument('files', help='Input files, will be concatenated', nargs='+')
    parser.add_argument('-o', '--outpath', help='Output path (if omitted)', default=argparse.SUPPRESS)
    args = parser.parse_args()

    inpaths = args.files
    indata = ''
    for inpath in inpaths:
        with open(inpath) as ifile:
            indata += ifile.read()
    if indata[-1] != '\n':
        indata += '\n'

    parser = MMKParser(indata)
    outstr = parser.parse()
    if outstr is None:
        exit(2)

    firstpath = inpaths[0]
    if 'o' in args:
        outpath = args.o
    elif 'outpath' in args:
        outpath = args.outpath
    else:
        if firstpath.rfind('.') == -1 or firstpath.rfind('.') < firstpath.rfind('/'):
            outpath = firstpath + '.out.txt'
        else:
            outpath = firstpath[:firstpath.rfind('.')] + '.out.txt'
    with open(outpath, 'w') as ofile:
        ofile.write(outstr)

    # subprocess.call('build.cmd')
