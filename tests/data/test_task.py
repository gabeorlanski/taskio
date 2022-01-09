"""
Tests for the Task features
"""
from transformers import AutoTokenizer
from datasets import Dataset

from pathlib import Path
from yamrf.data import Task, load_task_from_cfg


class TestTask:
    def test_read_data(self, tmpdir, simple_config, dummy_data):
        tmpdir_path = Path(tmpdir)
        task = load_task_from_cfg(simple_config)
        raw, tokenized = task.read_data(tmpdir_path, set_format="torch")

        actual = raw.to_dict()
        dummy_data['input_sequence'] = list(
            map('Generate Python: {}'.format, dummy_data['input'])
        )
        dummy_data['target'] = dummy_data['output']
        assert actual == dummy_data

        def tokenize(ex, idx):
            return {
                'idx'   : idx,
                'labels': task.tokenizer(ex['target'])['input_ids'],
                **task.tokenizer(ex['input_sequence'])
            }

        expected_tokenized = Dataset.from_dict(dummy_data)
        expected_tokenized = expected_tokenized.map(
            tokenize,
            with_indices=True,
            remove_columns=expected_tokenized.column_names
        )
        assert tokenized.to_dict() == expected_tokenized.to_dict()
