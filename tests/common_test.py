import pytest
from tio import common


@pytest.mark.parametrize('references,expected', [
    (["A", "B", "C"], [["A", "B", "C"]]),
    ([["A", "B", "C"], ["D", "E", "F"]], [["A", "B", "C"], ["D", "E", "F"]])
], ids=['SingleRef', 'MultiRef'])
def test_prepare_references_for_bleu(references, expected):
    assert common.prepare_references_for_bleu(references) == expected
