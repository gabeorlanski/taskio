import numpy as np
import torch
from datasets import Dataset
from functools import partial
import logging
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
from transformers import PreTrainedTokenizer, AutoTokenizer
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
        dataset = self.dataset_load_fn(split)

        def preprocess(example, idx):
            for fn in self.preprocessors:
                example = fn(example)
            return {"idx": idx, **example}

        def tokenize(example, idx):
            out = {"idx": idx, **self.tokenizer(example.pop("input_sequence"))}

            # We do not need the input sequence after tokenizing.
            target_tokenized = self.tokenizer(example.pop("target"))
            out.update(
                {
                    "labels": target_tokenized["input_ids"],
                }
            )
            return out

        preprocessed = dataset.map(
            preprocess,
            with_indices=True,
            num_proc=num_procs,
            remove_columns=dataset.column_names,
        )

        # Save the preprocessed under the split name so that later it can be
        # used to save aligned predictions after evaluation.
        self.preprocessed_splits[split] = preprocessed

        tokenized = preprocessed.map(
            tokenize,
            with_indices=True,
            num_proc=num_procs,
            remove_columns=dataset.column_names,
        )

        if set_format:
            tokenized.set_format(type=set_format)

        return tokenized

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

    def postprocess_np(
            self,
            predictions: np.ndarray,
            targets: np.ndarray
    ) -> Tuple[List[List[str]], List[str]]:
        return self.postprocess(
            predictions=torch.from_numpy(predictions),
            targets=torch.from_numpy(targets)
        )

    def postprocess(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Postprocess the raw predictions and the raw targets.

        Args:
            predictions (torch.Tensor): The raw predictions.
            targets (torch.Tensor): The raw targets.

        Returns:
            The list of predictions and the list of targets.
        """

        if predictions.size()[0] != targets.size()[0]:
            predictions_size_str = ', '.join(map(str, predictions.size()))
            target_size_str = ', '.join(map(str, targets.size()))
            logger.error("Predictions and targets do not have the same first size.")
            logger.error(f"Predictions has a size of [{predictions_size_str}]")
            logger.error(f"Targets has a size of [{target_size_str}]")
            raise ValueError(f"Sizes do not match. {predictions_size_str} != {target_size_str}")

        n_dims = len(predictions.size())
        num_sequences_per_sample = 1

        if n_dims > 3:
            logger.error(f"Postprocess cannot handle {len(predictions.size())} "
                         f"dimension tensors.")
            raise ValueError("Postprocess cannot handle predictions or targets "
                             "with more than 3 dimensions.")
        elif n_dims == 1:
            # In the case it is a 1-D array, reshape it into a (samples, 1)
            # tensor so that we can decode it with the tokenizer.
            predictions = predictions.unsqueeze(-1)
            targets = targets.unsqueeze(1)
        elif n_dims == 3:
            # In the case that it is a 3-D array, reshape it into a
            # (samples * num return sequences, sequence length) tensor.
            num_sequences_per_sample = predictions.size()[1]

            predictions = torch.flatten(predictions, start_dim=0, end_dim=1)

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
        for pred_idx in range(0, targets.size()[0], num_sequences_per_sample):
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
    ):
        task_cls = Task.by_name(name)
        return task_cls(
            tokenizer=tokenizer,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            metric_fns=metric_fns,
            additional_splits=additional_splits,
            **(additional_kwargs or {})
        )


def load_processors_from_cfg(cfg: DictConfig) -> Tuple[List[Callable], List[Callable]]:
    """
    Create the pre- and post- processors from a given config.

    Args:
        cfg (DictConfig): The config to use.

    Returns:
        Tuple[List[Callable], List[Callable]]: The created preprocessors and
            postprocessors.
    """
    logger.debug("Loading processors")
    preprocessors = []
    postprocessors = []

    if cfg.get("preprocessors") is not None:
        logger.debug("Preprocessors found")
        preprocessors = [
            partial(Preprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in cfg["preprocessors"].items()
        ]
    logger.info(f"{len(preprocessors)} preprocessors found")

    if cfg.get("postprocessors") is not None:
        logger.debug("Postprocessors found")
        postprocessors = [
            partial(Postprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in cfg["postprocessors"].items()
        ]
    logger.info(f"{len(postprocessors)} postprocessors found")
    return preprocessors, postprocessors


def load_task_from_cfg(cfg: DictConfig) -> Task:
    """
    Create a Task from a cfg

    Args:
        cfg (DictConfig): The config to use.

    Returns:
        Task: The created task object.
    """
    logger.info(f"Initializing task registered to name '{cfg['task']['name']}'")
    cfg_dict = OmegaConf.to_object(cfg["task"])
    preprocessors, postprocessors = load_processors_from_cfg(cfg)
    logger.info(f"Metrics are {cfg.get('metrics', [])}")
    metrics = [Metric.by_name(metric) for metric in cfg.get('metrics', [])]

    return Task.get_task(
        name=cfg["task"]["name"],
        tokenizer=AutoTokenizer.from_pretrained(cfg['model']),
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        metric_fns=metrics,
        **cfg_dict.get('args', {}),
    )
