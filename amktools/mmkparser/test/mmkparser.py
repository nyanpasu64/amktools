from amktools import mmkparser
from ruamel.yaml import YAML

yaml = YAML(typ='safe')

tuning_data = r'test.brr: $F0 $0F'
tuning = yaml.load(tuning_data)


# after adding ADSR: `%tune "test.brr" %adsr w x y z`


def test_instruments():
    in_str = '''
    #instruments
    {
    	%tune "test.brr" $8F $E0 $00
    }
    '''

    p = mmkparser.MMKParser(in_str, tuning)
    outstr = p.parse()

    print(outstr)
