from ruamel.yaml import YAML

from amktools import mmkparser

yaml = YAML(typ='safe')

tuning = yaml.load(r'test.brr: $F0 $0F')


def test_strip():
    in_str = '\n\n\n; foo\n; bar\n\n\n\n\n'
    p = mmkparser.MMKParser(in_str)
    out = p.parse()
    assert out == in_str.strip() + '\n'


def test_instruments():
    in_str = ''';
#instruments
{
    %tune "test.brr" $8F $E0 $00
}
'''

    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()

    assert outstr == ''';
#instruments
{
    "test.brr" $8F $E0 $00 $F0 $0F
}
'''


# after adding ADSR: `%tune "test.brr" %adsr w x y z`
# def test_adsr...
