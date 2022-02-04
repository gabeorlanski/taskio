"""
Tests for the Processors
"""
from copy import copy

import pytest
from tio.processor import Preprocessor, Postprocessor
from functools import partial


@pytest.fixture()
def sample_input():
    yield {
        "input_sequence": "I like your funny words,",
        "target"        : "Magic Man!"
    }


def test_add_prefix(sample_input):
    processor = Preprocessor.by_name('add-prefix')
    result = processor(copy(sample_input), prefix='Test')
    assert result == {
        "input_sequence": f"Test {sample_input['input_sequence']}",
        "target"        : sample_input['target']
    }


def test_lm_target(sample_input):
    processor = Preprocessor.by_name('lm-target')
    result = processor(copy(sample_input))
    expected = f"{sample_input['input_sequence']} {sample_input['target']}"
    assert result == {
        "input_sequence": expected,
        "target"        : expected
    }


def test_add_suffix(sample_input):
    processor = Preprocessor.by_name('add-suffix')
    result = processor(copy(sample_input), suffix='Test', add_space=True)
    assert result == {
        "input_sequence": f"{sample_input['input_sequence']} Test",
        "target"        : sample_input['target']
    }


def test_concat(sample_input):
    processor = Preprocessor.by_name('concat')
    result = processor(copy(sample_input))
    expected = f"{sample_input['input_sequence']} {sample_input['target']}"
    assert result == {
        "input_sequence": expected,
        "target"        : sample_input['target']
    }


def test_split_on_phrase(sample_input):
    processor = Postprocessor.by_name('split')
    result = processor(sample_input['input_sequence'], split_phrase='your')
    assert result == " funny words,"


def test_strip_sequence(sample_input):
    processor = Postprocessor.by_name('strip')
    result = processor(" HELLO ")
    assert result == "HELLO"
