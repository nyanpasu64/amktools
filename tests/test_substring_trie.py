import pygtrie
import pytest

from amktools.utils.substring_trie import StringSlice


def test_stringslice():
    x = StringSlice('aaaabbbbcccc', 8)

    assert x[0] == 'c'
    assert x[-1] == 'c'
    assert x[:] == 'cccc'
    assert x[-4:] == 'cccc'
    with pytest.raises(TypeError):
        x[None]


def test_substring_trie():
    trie = pygtrie.CharTrie()
    trie['cccc'] = True

    input = StringSlice('aaaabbbbcccc', 8)
    assert trie.longest_prefix(input).key == 'cccc'
