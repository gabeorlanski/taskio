"""
Tests for the Task features
"""
import pytest
from datasets import Dataset
from transformers import AutoTokenizer
from tio import Task


class TestTask:

    @pytest.mark.parametrize("split", ['train', 'val'])
    def test_get_split(self, simple_config, dummy_data, split):
        task = Task.from_dict(
            simple_config['task'],
            AutoTokenizer.from_pretrained(simple_config['model'])
        )
        tokenized = task.get_split(split, set_format="torch")
        raw = task.preprocessed_splits[split]

        actual = raw.to_dict()
        dummy_data['input_sequence'] = dummy_data['input']
        dummy_data['target'] = dummy_data['output']
        assert actual == dummy_data

        def tokenize(ex, idx):
            label_data = task.tokenizer(ex['target'])
            return {
                'idx'   : idx,
                'labels': label_data['input_ids'],
                **task.tokenizer(ex['input_sequence'])
            }

        expected_tokenized = Dataset.from_dict(dummy_data)
        expected_tokenized = expected_tokenized.map(
            tokenize,
            with_indices=True,
            remove_columns=expected_tokenized.column_names
        )
        assert tokenized.to_dict() == expected_tokenized.to_dict()

    @pytest.mark.parametrize('sequences', [
        ['Do You Want Ants'],
        ['Because that', 'is how you get ants'],
        ["?"]
    ], ids=['Single', 'Double', "SingleToken"])
    def test_postprocess(self, simple_config, sequences):
        task = Task.from_dict(
            simple_config['task'],
            AutoTokenizer.from_pretrained(simple_config['model'])
        )
        task.postprocessors.append(lambda x: f"Test: {x}")

        sequence_tokenized = task.tokenizer(
            sequences,
            return_tensors='pt',
            padding='longest'
        )['input_ids']
        sequence_tokenized = sequence_tokenized

        expected = [f'Test: {p}' for p in sequences]

        actual = task.postprocess(sequence_tokenized)
        assert actual == expected

    def test_evaluate(self, simple_config):
        task = Task.from_dict(
            simple_config['task'],
            AutoTokenizer.from_pretrained(simple_config['model'])
        )
        task.metric_fns = [
            lambda preds, targets: {'em': sum(p == t for p, t in zip(preds, targets)) / len(preds)}
        ]

        result = task.evaluate(
            [["A"], ["B"], ["C"], ["E"], ["E"]],
            ["A", "B", "C", "D", "E"]
        )
        assert result == {
            "em": 0.8
        }
