import numpy as np
import torch
from datasets import Dataset
from functools import partial
import logging
import os
from pathlib import Path
from transformers import PreTrainedTokenizer
from typing import Dict, List, Callable, Tuple, Optional, Union

from tio.metrics import Metric
from tio.processor import Preprocessor, Postprocessor
from tio.registrable import Registrable

logger = logging.getLogger(__name__)
PathType = Union[os.PathLike, str, Path]


class Task(Registrable):
    """
    Base class for a task.

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer.

        preprocessors (List[Callable]): List of callables for preprocessing.
            Each must take in a single argument of type ``Dict`` and return
            a ``Dict``.

            Each ``example`` passed to the preprocessors will have an input
            sequence and a target entry.

        postprocessors (List[Callable]): List of callables for postprocessing.
            Each must take in a single argument of type ``Dict`` and return
            a ``Dict``.

        metric_fns (List[Callable]): List of metric functions to run on the
            postprocessed data. Each function must have the signature
            ``prediction, target``.

        additional_splits (Dict[str,PathType]): Dict of additional splits to
            add to SPLIT_MAPPING. Does NOT overwrite existing splits even if
            they share the same split name.

    """

    SPLIT_MAPPING = {}

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            additional_splits: Dict[str, PathType] = None
    ):
        self.input_sequence_key = ("input_sequence",)
        self.target_key = "target"
        self.tokenizer = tokenizer
        self.preprocessors = [self.map_to_standard_entries, *(preprocessors or [])]
        self.postprocessors = postprocessors or []
        self.preprocessed_splits = {}
        self.metric_fns = metric_fns or []

        if additional_splits:
            logger.debug(f"Adding {len(additional_splits)} additional splits")
            for split_name, value in additional_splits.items():
                if split_name in self.SPLIT_MAPPING:
                    logger.warning(f"Trying to add split {split_name}, but "
                                   f"already in SPLIT_MAPPING with value "
                                   f"{self.SPLIT_MAPPING[split_name]}")
                    continue

                logger.debug(f"Adding split {split_name} with value {value}")
                self.SPLIT_MAPPING[split_name] = value

    def dataset_load_fn(self, split: str) -> Dataset:
        """
        Method to read in the raw data that is to be implemented by subclasses.
        Args:
            split (str): The split to use.

        Returns:
            Dataset: The processed HuggingFace Dataset.

        """
        raise NotImplementedError()

    @staticmethod
    def map_to_standard_entries(sample: Dict) -> Dict:
        """
        Function that must be implemented by sub-classes for mapping dataset
        specific columns to standardized ones.

        The output dict must have the keys ``"input_sequence"`` and
        ``"target"``.

        Args:
            sample (Dict): The dict for a given sample in the dataset.

        Returns:
            Dict: The sample with the added standard entries.

        """
        raise NotImplementedError()

    def get_split(
            self, split: str, num_procs: int = 1, set_format: Optional[str] = None
    ) -> Dataset:
        """
        Method to read and preprocess dataset.

        It returns the tokenized dataset and saves the preprocessed dataset
        internally under the split name

        Args:
            split (str): The split to use.
            num_procs (int): Number of processes to use in preprocessing.
            set_format (Optional[str]): If passed, the tokenized dataset will
                be set to this format.

        Returns:
            Dataset: The preprocessed and tokenized dataset.
        """

        def tokenize(example, idx):
            # We do not pop so that we can still remove the columns later.
            out = {"idx": idx, **self.tokenizer(example["input_sequence"])}

            target_tokenized = self.tokenizer(example["target"])
            out.update(
                {
                    "labels": target_tokenized["input_ids"],
                }
            )
            return out

        preprocessed = self.preprocess(split, num_procs)

        # Save the preprocessed under the split name so that later it can be
        # used to save aligned predictions after evaluation.
        self.preprocessed_splits[split] = preprocessed

        tokenized = preprocessed.map(
            tokenize,
            with_indices=True,
            num_proc=num_procs,
            remove_columns=preprocessed.column_names,
        )

        if set_format:
            tokenized.set_format(type=set_format)

        return tokenized

    def _preprocess(self, example: Dict, idx: int) -> Dict:
        """
        Preprocess a single example.

        Args:
            example (Dict): The example to preprocess
            idx (int): The idx of the example.

        Returns:
            Dict: The preprocessed example.
        """
        for fn in self.preprocessors:
            example = fn(example)
        return {"idx": idx, **example}

    def preprocess(self, split: str, num_procs: int = 1) -> Dataset:
        """
        Preprocess a split.

        Args:
            split (str): The split to preprocess. Must be in ``SPLIT_MAPPING``
            num_procs (int): Number of processes to use.

        Returns:
            Dataset: The preprocessed split.
        """
        dataset = self.dataset_load_fn(split)

        return dataset.map(
            self._preprocess,
            with_indices=True,
            num_proc=num_procs,
            remove_columns=dataset.column_names,
        )

    def postprocess(
            self,
            predictions: np.ndarray,
            targets: np.ndarray
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Postprocess the raw predictions and the raw targets.

        Args:
            predictions (np.ndarray): The raw predictions.
            targets (np.ndarray): The raw targets.

        Returns:
            The list of predictions and the list of targets.
        """

        if predictions.shape[0] != targets.shape[0]:
            predictions_size_str = ', '.join(map(str, predictions.size()))
            target_size_str = ', '.join(map(str, targets.size()))
            logger.error("Predictions and targets do not have the same first size.")
            logger.error(f"Predictions has a size of [{predictions_size_str}]")
            logger.error(f"Targets has a size of [{target_size_str}]")
            raise ValueError(f"Sizes do not match. {predictions_size_str} != {target_size_str}")

        n_dims = len(predictions.shape)
        num_sequences_per_sample = 1

        if n_dims > 3:
            logger.error(f"Postprocess cannot handle {len(predictions.size())} "
                         f"dimension tensors.")
            raise ValueError("Postprocess cannot handle predictions or targets "
                             "with more than 3 dimensions.")
        elif n_dims == 1:
            # In the case it is a 1-D array, reshape it into a (samples, 1)
            # tensor so that we can decode it with the tokenizer.
            predictions = np.expand_dims(predictions, axis=-1)
            targets = np.expand_dims(targets, 1)
        elif n_dims == 3:
            # In the case that it is a 3-D array, reshape it into a
            # (samples * num return sequences, sequence length) tensor.
            num_sequences_per_sample = predictions.shape[1]

            predictions = predictions.reshape(predictions.shape[0] * predictions.shape[1], -1)

        # Decode both the predictions and the targets
        def postprocess(sequence):
            for fn in self.postprocessors:
                sequence = fn(sequence)
            return sequence

        predictions_decoded = list(map(postprocess, self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )))
        targets_decoded = list(map(postprocess, self.tokenizer.batch_decode(
            targets,
            skip_special_tokens=True
        )))

        out_preds = []
        for pred_idx in range(0, predictions.shape[0], num_sequences_per_sample):
            out_preds.append(
                predictions_decoded[pred_idx:pred_idx + num_sequences_per_sample]
            )

        return out_preds, targets_decoded

    def evaluate(
            self,
            predictions: List[List[str]],
            targets: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against targets with the metrics.

        Args:
            predictions (List[List[str]]): The predictions. For evaluation, it
                will use the first string of each nested list.
            targets (List[str]): The targets.

        Returns:
            The metrics dict.
        """
        metrics = {}
        preds_single = [p[0] for p in predictions]
        for metric in self.metric_fns:
            metrics.update(metric(preds_single, targets))

        return metrics

    @classmethod
    def get_task(
            cls,
            name,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            additional_splits: Dict[str, PathType] = None,
            additional_kwargs: Dict = None
    ) -> 'Task':
        """
        Get and instantiate a task by name.
        Args:
            name (str): The task to load.

            tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer.

            preprocessors (List[Callable]): List of callables for preprocessing.
                Each must take in a single argument of type ``Dict`` and return
                a ``Dict``.

                Each ``example`` passed to the preprocessors will have an input
                sequence and a target entry.

            postprocessors (List[Callable]): List of callables for postprocessing.
                Each must take in a single argument of type ``Dict`` and return
                a ``Dict``.

            metric_fns (List[Callable]): List of metric functions to run on the
                postprocessed data. Each function must have the signature
                ``prediction, target``.

            additional_splits (Dict[str,PathType]): Dict of additional splits to
                add to SPLIT_MAPPING. Does NOT overwrite existing splits even if
                they share the same split name.

            additional_kwargs (Dict): Other kwargs to pass to the constructor.

        Returns:
            Task: The initialized task.
        """
        task_cls = Task.by_name(name)
        return task_cls(
            tokenizer=tokenizer,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            metric_fns=metric_fns,
            additional_splits=additional_splits,
            **(additional_kwargs or {})
        )

    @classmethod
    def from_dict(cls, task_dict: Dict, tokenizer: PreTrainedTokenizer) -> 'Task':
        """
        Load a task from a dictionary.

        It expects the keys:

            name (str): The task to load.

            preprocessors (List[Callable]): List of callables for preprocessing.
                    Each must take in a single argument of type ``Dict`` and return
                    a ``Dict``.

                    Each ``example`` passed to the preprocessors will have an input
                    sequence and a target entry.

            postprocessors (List[Callable]): List of callables for postprocessing.
                Each must take in a single argument of type ``Dict`` and return
                a ``Dict``.

            metric_fns (List[Callable]): List of metric functions to run on the
                postprocessed data. Each function must have the signature
                ``prediction, target``.

            additional_splits (Dict[str,PathType]): Dict of additional splits to
                add to SPLIT_MAPPING. Does NOT overwrite existing splits even if
                they share the same split name.

            additional_kwargs (Dict): Other kwargs to pass to the constructor.

        Args:
            task_dict (Dict): The dict to use.

            tokenizer (PreTrainedTokenizer): The tokenizer to use.

        Returns:
            Task: The initialized task.
        """
        preprocessors = [
            partial(Preprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in task_dict.get("preprocessors", {}).items()
        ]
        postprocessors = [
            partial(Postprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in task_dict.get("postprocessors", {}).items()
        ]
        metrics = [Metric.by_name(metric) for metric in task_dict.get('metrics', [])]

        return Task.get_task(
            name=task_dict['name'],
            tokenizer=tokenizer,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            metric_fns=metrics,
            additional_splits=task_dict.get('additional_splits'),
            additional_kwargs=task_dict.get("additional_kwargs")
        )
#
