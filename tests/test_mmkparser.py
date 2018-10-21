import io
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from amktools import mmkparser


# Util testing


def test_regex():
    string = 'a' * 16 + 'cb'

    bc = mmkparser.any_of('bc')
    match = bc.search(string)
    assert match
    assert match.start() == match.end() == 16

    na = mmkparser.none_of('a')
    match = na.search(string)
    assert match
    assert match.start() == match.end() == 16


yaml = YAML(typ='safe')

tuning = yaml.load(r'test.brr: $F0 $0F')


# MMKParser file path tests


def test_constants() -> None:
    assert mmkparser.RETURN_ERR != 0


def call_mmkparser(filename: Path, expected_ret: int) -> None:
    filename.touch()
    ret = mmkparser.main([str(filename)])
    assert ret == expected_ret


txt = Path('file.txt')
mmk = Path('file.mmk')
parse_output = 'parse_output'


def test_dont_overwrite_txt(mocker) -> None:
    """ Ensures that mmkparser returns an error, instead of overwriting
    foo.txt supplied as input. """

    mocker.patch.object(mmkparser.MMKParser, 'parse')
    mmkparser.MMKParser.parse.return_value = parse_output

    with CliRunner().isolated_filesystem():
        call_mmkparser(txt, mmkparser.RETURN_ERR)


def test_extension(mocker) -> None:
    """ Ensures that mmkparser parses foo.mmk to foo.txt."""

    mocker.patch.object(mmkparser.MMKParser, 'parse')
    mmkparser.MMKParser.parse.return_value = parse_output

    with CliRunner().isolated_filesystem():
        for i in range(2):
            # We should overwrite output txt files without issues.
            call_mmkparser(mmk, 0)
            assert txt.exists()
            with txt.open() as f:
                assert f.read() == parse_output


# Input string processing


def test_define():
    in_str = '''\
;
%define x 1
%x
'''
    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()
    assert outstr == '''\
;

1
'''


@pytest.mark.xfail(strict=True)
def test_define_failed():
    in_str = '''\
;
%define x 1
%define y v%x
%y
'''
    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()
    assert outstr == '''\
;


v1
'''


def test_notelen():
    in_str = '''\
c4 c2 c1 c=48
%notelen on

; All notes/etc with durations.
c1 d1 e1 f1 g1 a1 b1 r1 ^1

; default length command
l1 c

c1c2c4c5
c/2c/3c/4c/6c/8 c/48
c2/3c3/4 c47/48
c=47

%notelen off
c4 c2 c1 c=48
'''
    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()
    assert outstr == '''\
c4 c2 c1 c=48


; All notes/etc with durations.
c4 d4 e4 f4 g4 a4 b4 r4 ^4

; default length command
l4 c

c4c2c1c=240
c8c12c16c24c32 c192
c6c=36 c=47
c=47


c4 c2 c1 c=48
'''


def test_instruments():
    in_str = '''#instruments
{
    %tune "test.brr" $8F $E0 $00
    "test.brr" %adsr -1,-1,full,0 $01 $23
    %tune "test.brr" %adsr -1,-1,full,0
}
'''
    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()
    assert outstr.lower() == '''#instruments
{
    "test.brr" $8f $e0 $00 $f0 $0f
    "test.brr" $ff $e0 $a0 $01 $23
    "test.brr" $ff $e0 $a0 $f0 $0f
}
'''


def test_instruments_comments():
    """ %tune needs to stop before reaching comments or something. maybe
    trim off all trailing space and append after tuning bytes?"""

    in_str = '''\
#instruments
{
    %tune "test.brr" $8F $E0 $00    ; foo
}
'''
    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()

    # Ideally I'd put the spaces *after* the tuning, but that's hard.
    assert outstr.lower() == '''\
#instruments
{
    "test.brr" $8f $e0 $00     $f0 $0f; foo
}
'''


# TODO parameterize
def test_commands():
    in_str = ''';
%vbend,4,255\t
%ybend 4 20\t

%gain,direct,$73
%gain,set,$73
%gain,down,$03
%gain,exp,$03
%gain,up,$03
%gain,bent,$03
;
'''
    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()
    assert outstr.lower() == ''';
$e8 $c0 $ff\t
$dc $c0 $14\t

$fa $01 $73
$fa $01 $73
$fa $01 $83
$fa $01 $a3
$fa $01 $c3
$fa $01 $e3
;
'''


def test_sweep():
    def metadata():
        return mmkparser.WavetableMetadata(
            nsamp=64,
            ntick=4,
            fps=None,
            wave_sub=2,
            env_sub=1,
            pitches=[60.0, 60.5, 61.0, 61.5],
        )
    meta_dict = {
        'untrunc': metadata(),
        'truncSilent': metadata(),
    }
    in_str = '''\
#samples
{
	%silent "silent.brr"
	%wave_group "untrunc"; make sure comments work
	%wave_group "truncSilent" 1 silent
}
#instruments
{
	%instr "silent.brr" $00 $00 $00 $01 $00
	%wave_group "untrunc" $00 $00 $00; make sure comments work
	%wave_group "truncSilent" $00 $00 $00
}
#0
%wave_sweep "untrunc" 96
%wave_sweep "truncSilent" 96
%wave_sweep "untrunc" 1
%wave_sweep "truncSilent" 1
; avert crash on eof
'''
    p = mmkparser.MMKParser(in_str, tuning, meta_dict)
    outstr = p.parse()

    # %wave_group produces "ugly" missing indentation, so ignore all whitespace.
    words = outstr.lower().split()
    tune_val = f'${metadata().nsamp//16:02} $00'
    assert words == '\n'.join([f'''\
#samples
{{
	"silent.brr"
	"untrunc-000.brr"; make sure comments work
	"untrunc-001.brr"
	"truncSilent-000.brr"
}}
#instruments
{{
	"silent.brr" $00 $00 $00 $01 $00
	"untrunc-000.brr" $00 $00 $00 {tune_val}; make sure comments work
	"untrunc-001.brr" $00 $00 $00 {tune_val}
	"truncSilent-000.brr" $00 $00 $00 {tune_val}
}}
#0''',

# %wave_sweep "untrunc" 96
# ADSR        silent.brr+tune    legato
'$ED $7d $e0     $f3 $00 $04     $F4 $01',
#   smp[#4]=01 detune=00       detune=80
''' $f6 $04 $01  $ee $00  o4c=1  $ee $80  o4c=1
    $f6 $04 $02  $ee $00  o4c+=1 $ee $80  o4c+=93''',
# unlegato, detune=0
'''
$F4 $01     $ee $00''',

# %wave_sweep "truncSilent" 96
'''
$ED $7d $e0  $f3 $00 $04    $F4 $01
    $f6 $04 $03  $ee $00  o4c=1''',
#   GAIN fadeout
''' $FA $01 $98  o4c=95
$F4 $01     $ee $00''',

# %wave_sweep "untrunc" 1
'''$ED $7d $e0  $f3 $00 $04    $F4 $01
    $f6 $04 $01  $ee $00  o4c=1
$F4 $01     $ee $00''',

# %wave_sweep "truncSilent" 1
'''
$ED $7d $e0  $f3 $00 $04    $F4 $01
    $f6 $04 $03  $ee $00  o4c=1
$F4 $01     $ee $00

; avert crash on eof
''']).lower().split()
#$ED $7d $e0  $f3 $00 $04    $F4 $01  $f6 $04 $01  $ee $00  o4c=1  $ee $80  o4c=1  $f6 $04 $02  $ee $81  o4d=1
#ADSR         silent.brr+tune LEGATO  smp#[ch]=xx  detune00 note



def test_terminator():
    """ Ensures commands (words) can be terminated by ] without whitespace. """
    in_str = '[%gain direct $03]\n'
    p = mmkparser.MMKParser(in_str, None)
    out = p.parse()
    assert out.lower() == '[$fa $01 $03]\n'


# after adding ADSR: `%tune "test.brr" %adsr w x y z`
# def test_adsr...

# Edge cases and error-handling

def test_strip():
    in_str = '\n\n\n; foo\n; bar\n\n\n\n\n'
    p = mmkparser.MMKParser(in_str, None)
    out = p.parse()
    assert out == in_str.strip() + '\n'


def test_error():
    in_str = '%asdfasdf ; ; ; ; ; ; ; ; ;'
    with mock.patch('sys.stderr', new=io.StringIO()) as fake:  # type: io.StringIO
        p = mmkparser.MMKParser(in_str, None)
        with pytest.raises(mmkparser.MMKError):
            p.parse()

    val = fake.getvalue()  # type: str
    assert 'asdfasdf' in val


def test_error_at_eof():
    in_str = '%asdfasdf'
    with mock.patch('sys.stderr', new=io.StringIO()) as fake:  # type: io.StringIO
        p = mmkparser.MMKParser(in_str, None)
        with pytest.raises(mmkparser.MMKError):
            p.parse()

    val = fake.getvalue()  # type: str
    assert 'asdfasdf' in val


def test_eof():
    in_str = '%reset'
    p = mmkparser.MMKParser(in_str, None)
    p.parse()


@pytest.mark.xfail(reason='parse_int() causes error at EOF', strict=True)
def test_command_eof():
    in_str = 'v128'
    p = mmkparser.MMKParser(in_str, None)
    p.parse()
