from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURES_ROOT = PROJECT_ROOT.joinpath("test_fixtures")

# Check if we are in the home dir of the repo.
assert PROJECT_ROOT.joinpath("LICENSE.md").exists()


def prepare_references_for_bleu(references: List):
    if isinstance(references[0], list):
        references = [[t for t in target] for target in references]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        references = [references]
    return references
