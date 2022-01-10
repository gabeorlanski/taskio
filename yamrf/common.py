from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURES_ROOT = PROJECT_ROOT.joinpath("test_fixtures")

# Check if we are in the home dir of the repo.
assert PROJECT_ROOT.joinpath("LICENSE.md").exists()
