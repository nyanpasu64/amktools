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


# Functionality


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
