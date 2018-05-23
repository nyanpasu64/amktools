import pytest
from ruamel.yaml import YAML

from amktools import mmkparser
from amktools.mmkparser import MMKError

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
    p = mmkparser.MMKParser(in_str, None)
    with pytest.raises(MMKError):
        p.parse()


def test_error_at_eof():
    in_str = '%asdfasdf'
    p = mmkparser.MMKParser(in_str, None)
    with pytest.raises(MMKError):
        p.parse()


def test_eof():
    in_str = '%reset'
    p = mmkparser.MMKParser(in_str, None)
    p.parse()
