from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURES_ROOT = PROJECT_ROOT.joinpath("test_fixtures")

# Check if we are in the home dir of the repo.
assert PROJECT_ROOT.joinpath("LICENSE.md").exists()


def prepare_inputs_for_bleu(inputs: List):
    if not isinstance(inputs[0], list):
        inputs = [[t for t in target] for target in inputs]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        inputs = [inputs]
    return inputs
