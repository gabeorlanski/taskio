import pytest
from datasets import Dataset
from pathlib import Path
from typing import Dict
from copy import deepcopy
from tio import Task

__all__ = [
    "DummyTask"
]

EXPECTED_DUMMY_DATA = {
    "idx"   : [0, 1, 2],
    "input" : ["The comment section is ", "The butcher of ", "Get "],
    "output": ["out of control.", "Blevkin.", "Some."]
}


@pytest.fixture()
def dummy_data():
    yield deepcopy(EXPECTED_DUMMY_DATA)


@Task.register('dummy')
class DummyTask(Task):
    SPLIT_MAPPING = {
        "train": deepcopy(EXPECTED_DUMMY_DATA),
        "val"  : deepcopy(EXPECTED_DUMMY_DATA)
    }

    @staticmethod
    def map_to_standard_entries(sample: Dict) -> Dict:
        sample['input_sequence'] = sample['input']
        sample['target'] = sample['output']
        return sample

    def dataset_load_fn(self, split: str) -> Dataset:
        return Dataset.from_dict(EXPECTED_DUMMY_DATA)
