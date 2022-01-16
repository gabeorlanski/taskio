import logging
from typing import Dict

from tio.registrable import Registrable

logger = logging.getLogger(__name__)


class Preprocessor(Registrable):
    """
    Just a wrapper for the registrable so that preprocessing
    functions can be registered.
    """

    pass


class Postprocessor(Registrable):
    """
    Just a wrapper for the registrable so that postprocessing functions can be
    registered.
    """

    pass


@Preprocessor.register("add-prefix")
def add_prefix(example: Dict, prefix: str):
    example["input_sequence"] = f"{prefix} {example['input_sequence']}"
    return example


@Preprocessor.register('lm-target')
def lm_target(example: Dict):
    example['target'] = f'{example["input_sequence"]} {example["target"]}'
    return example


@Preprocessor.register('add-suffix')
def add_suffix(example: Dict, suffix: str, key: str):
    example[key] = f"{example[key]} {suffix}"
    return example


@Preprocessor.register('concat')
def concat(ex):
    ex['input_sequence'] = f"{ex['input_sequence']} {ex['target']}"
    ex['target'] = ex['input_sequence']
    return ex
