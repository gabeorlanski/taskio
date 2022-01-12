import pytest
import yaml
from tio.common import FIXTURES_ROOT
from .dummy_objects import *
from omegaconf import OmegaConf


@pytest.fixture()
def tiny_model_name():
    yield "patrickvonplaten/t5-tiny-random"


@pytest.fixture()
def simple_config():
    yield OmegaConf.create(yaml.load(
        FIXTURES_ROOT.joinpath('configs', 'simple.yaml').open('r', encoding='utf-8'),
        yaml.Loader
    ))
