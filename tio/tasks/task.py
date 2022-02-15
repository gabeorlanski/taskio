from collections import defaultdict

import numpy as np
from datasets import Dataset, IterableDataset
from functools import partial
import logging
import os
from pathlib import Path
from transformers import PreTrainedTokenizer
from typing import Dict, List, Callable, Tuple, Optional, Union, Generator

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

        split_mapping (Dict[str,PathType]): Dict of additional splits to
            add to SPLIT_MAPPING. Does NOT overwrite existing splits even if
            they share the same split name.

    """

    SPLIT_MAPPING = {}

    # Iterable datasets do not have column names, so use this to remove columns
    # from the preprocessed split.
    RAW_COLUMN_NAMES = []

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            split_mapping: Dict[str, PathType] = None
    ):
        self.input_sequence_key = ("input_sequence",)
        self.target_key = "target"
        self.tokenizer = tokenizer
        self.preprocessors = [self.map_to_standard_entries, *(preprocessors or [])]
        self.postprocessors = postprocessors or []
        self.preprocessed_splits = {}
        self.metric_fns = metric_fns or []

        if split_mapping:
            logger.debug(f"Adding {len(split_mapping)} additional splits")
            for split_name, value in split_mapping.items():
                if split_name in self.SPLIT_MAPPING:
                    logger.warning(f"Trying to add split {split_name}, but "
                                   f"already in SPLIT_MAPPING with value "
                                   f"{self.SPLIT_MAPPING[split_name]}")
                    continue

                logger.debug(f"Adding split {split_name} with value {value}")
                self.SPLIT_MAPPING[split_name] = value

    def _load_samples(self, split: str) -> Dataset:
        """
        Read in the raw data that is to be implemented by subclasses.



        Args:
            split (str): The split to use.

        Returns:
            Dataset: The dataset.

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
            self,
            split: str,
            num_procs: int = 1,
            set_format: Optional[str] = None,
            add_special_tokens: bool = True,
            overwrite_cache: bool = False
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
            add_special_tokens (bool): Add special tokens with the tokenizer.
                Default is True.
            overwrite_cache (bool): Overwrite HuggingFace's cache.

        Returns:
            Dataset: The preprocessed and tokenized dataset.
        """

        def tokenize(example, idx):
            # We do not pop so that we can still remove the columns later.
            out = {
                "idx": idx, **self.tokenizer(example["input_sequence"],
                                             add_special_tokens=add_special_tokens)
            }

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
            load_from_cache_file=not overwrite_cache,
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
        dataset = self._load_samples(split)

        return dataset.map(
            self._preprocess,
            with_indices=True,
            num_proc=num_procs,
            remove_columns=dataset.column_names,
        )

    def postprocess(self, sequence):
        for fn in self.postprocessors:
            sequence = fn(sequence)
        return sequence

    def postprocess_raw_tokens(
            self,
            sequences: np.ndarray
    ) -> List[str]:
        """
        Postprocess the raw predictions and the raw targets.

        Args:
            sequences (np.ndarray): The raw sequences to process.

        Returns:
            The list of processed sequences.
        """

        return list(map(self.postprocess, self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True
        )))

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
            split_mapping: Dict[str, PathType] = None,
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

            split_mapping (Dict[str,PathType]): Dict of additional splits to
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
            split_mapping=split_mapping,
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
        metrics = []
        for metric in task_dict.get('metrics', []):
            if isinstance(metric, dict):
                metric_name, metric_dict = list(metric.items())
            else:
                metric_name = metric
                metric_dict = {}
            metrics.append(Metric.from_dict(metric_name, metric_dict))

        return Task.get_task(
            name=task_dict['name'],
            tokenizer=tokenizer,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            metric_fns=metrics,
            split_mapping=task_dict.get('split_mapping'),
            additional_kwargs=task_dict.get("kwargs")
        )

    def serialize_task_features(
            self,
            idx: int,
            predictions: List,
            processed_sample: Dict
    ) -> Dict:
        """
        Function for serializing task specific features. Subclasses MUST 
        implement this.

        Example of when this is useful: Returning columns that are not used in
        prediction but you still want to save them.
        
        This should NOT return the following keys as they are handled by
        serialize_predictions:

        * 'idx'
        * 'target'
        * 'input_sequence'
        * 'prediction'
        
        Args:
            idx (int): The index of the prediction. 
            predictions (List[str]): The list of predictions. 
            processed_sample (Dict): The processed Sample. 

        Returns:
            serialized_features (Dict): The serialized task specific featueres.

        """
        raise NotImplementedError()

    def serialize_predictions(
            self,
            split: str,
            indices: List,
            predictions: List[List]
    ) -> Generator[Dict, None, None]:
        """
        Serialize a prediction to a dict.

        Args:
            split (str): The split the predictions came from.
            indices (List): The indices corresponding to the predictions.
            predictions (List[List]): The list of predictions for each sample.

        Returns:
            A generator of dicts for each sample.
        """

        processed_data = self.preprocessed_splits[split]

        assert len(indices) == len(predictions), "Indices must be the same length as predictions"

        for idx, preds in zip(indices, predictions):
            processed_sample = processed_data[idx]
            sample = self.serialize_task_features(idx, preds, processed_sample)
            yield {
                'idx'           : idx,
                'target'        : processed_sample['target'],
                'input_sequence': processed_sample['input_sequence'],
                'prediction'    : preds,
                **sample
            }
