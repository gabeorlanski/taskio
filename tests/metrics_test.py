"""
Tests for the Metrics
"""
import pytest
from tio import metrics


@pytest.mark.parametrize("preds,targets,expected", [
    (["A", "B", "C"], ["A", "B", "C"], {'em': 100}),
    (["A", "B", "C"], ["F", "E", "F"], {'em': 0}),
], ids=['AllMatch', "NoMatch"])
def test_exact_match(preds, targets, expected):
    metric = metrics.Metric.from_dict('exact-match', {})
    result = metric(preds, targets)
    assert set(result) == {'em'}
    assert result['em'] == pytest.approx(expected['em'])


@pytest.mark.parametrize("preds,targets,expected", [
    (["this is a test", "this is not a test"], ["this is a test", "this is not a test"],
     {'bleu': 100}),
    (["this is a test", "this is not a test"], ["No", "This Is Patrick"], {'bleu': 0}),
], ids=['AllMatch', "NoMatch"])
def test_bleu(preds, targets, expected):
    metric = metrics.Metric.from_dict('bleu', {})
    result = metric(preds, targets)
    assert set(result) == {'bleu'}
    assert result['bleu'] == pytest.approx(expected['bleu'])
