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

    @pytest.mark.parametrize('predictions',
                             [['Do You Want Ants'],
                              ['Because that ', 'is how you get ants'],
                              "a"],
                             ids=['SinglePred', 'DoublePred', "SingleToken"])
    def test_postprocess(self, simple_config, predictions):
        task = Task.from_dict(
            simple_config['task'],
            AutoTokenizer.from_pretrained(simple_config['model'])
        )
        task.postprocessors.append(lambda x: f"Test: {x}")
        if isinstance(predictions, str):
            preds_tokked = task.tokenizer(
                [predictions], add_special_tokens=False, return_tensors='pt'
            )['input_ids']
            preds_tokked = preds_tokked.squeeze(0)
            preds_input = preds_tokked
            target_tokenized = task.tokenizer(
                ["b"], add_special_tokens=False, return_tensors='pt'
            )['input_ids']
            target_tokenized = target_tokenized.squeeze(0)
        else:
            target = ["Archer"]
            target_tokenized = task.tokenizer(target, return_tensors='pt')['input_ids']
            preds_tokked = task.tokenizer(
                predictions,
                return_tensors='pt',
                padding='longest'
            )['input_ids']
            if len(predictions) == 2:
                preds_input = preds_tokked.unsqueeze(0)
            else:
                preds_input = preds_tokked

        expected_preds = list(
            map(
                lambda x: f"Test: {task.tokenizer.decode(x, skip_special_tokens=True)}",
                preds_tokked
            )
        )
        expected_targets = list(
            map(
                lambda x: f"Test: {task.tokenizer.decode(x, skip_special_tokens=True)}",
                target_tokenized
            )
        )

        if isinstance(predictions, list):
            expected_preds = [expected_preds]
        else:
            expected_preds = [[pred] for pred in expected_preds]

        actual_preds, actual_targets = task.postprocess_np(preds_input.numpy(),
                                                           target_tokenized.numpy())
        assert actual_preds == expected_preds
        assert actual_targets == expected_targets

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
