import io
from unittest import mock

import pytest
from ruamel.yaml import YAML

from amktools import mmkparser
from amktools.mmkparser import MMKError, any_of, none_of


# Util testing

# def test_find_first():
#     s = 'fasdffsdfdsafsadsfsdafsdaf'
#
#     assert find_first(s, '!') is None
#     assert find_first(s, '!', 2, 1) is None
#
#     s = 'aaaaaaaaaaaaaaaaa'
#     assert find_first(s, '!', 2, 1) is None


def test_regex():
    string = 'a' * 16 + 'cb'

    bc = any_of('bc')
    match = bc.search(string)
    assert match
    assert match.start() == match.end() == 16

    na = none_of('a')
    match = na.search(string)
    assert match
    assert match.start() == match.end() == 16


yaml = YAML(typ='safe')

tuning = yaml.load(r'test.brr: $F0 $0F')


# Functionality

def test_instruments():
    in_str = '''#instruments
{
    %tune "test.brr" $8F $E0 $00
}
'''
    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()
    assert outstr == '''#instruments
{
    "test.brr" $8F $E0 $00 $F0 $0F
}
'''


def test_commands():
    in_str = ''';
%vbend,4,255
'''
    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()
    assert outstr.lower() == ''';
$e8 $c0 $ff
'''


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
        with pytest.raises(MMKError):
            p.parse()

    val = fake.getvalue()  # type: str
    assert 'asdfasdf' in val


def test_error_at_eof():
    in_str = '%asdfasdf'
    with mock.patch('sys.stderr', new=io.StringIO()) as fake:  # type: io.StringIO
        p = mmkparser.MMKParser(in_str, None)
        with pytest.raises(MMKError):
            p.parse()

    val = fake.getvalue()  # type: str
    assert 'asdfasdf' in val


def test_eof():
    in_str = '%reset'
    p = mmkparser.MMKParser(in_str, None)
    p.parse()
