from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sacrebleu

from tio.registrable import Registrable

from tio.common import prepare_references_for_bleu, PROJECT_ROOT

PATH_TO_HERE = Path(__file__).parent.resolve()


class Metric(Registrable):
    def __init__(self, name, main_metric_key):
        self.name = name
        self.main_metric_key = main_metric_key

    def __call__(self, predictions: List[str], targets: List[str]) -> Dict:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, name, metric_dict):
        if name not in Registrable._registered_components[cls]:
            raise KeyError(f"{name} is not a valid metric")
        metric_cls = Registrable._registered_components[cls][name]
        return metric_cls(**metric_dict)

    def calculate_single_example(self, prediction: str, target: str) -> Dict:
        """
        Wrapper for the self __call__ function for handling a single example.
        Useful for metrics like BLEU where sentence bleu and corpus bleu are
        not the same thing.

        Args:
            prediction (str): The prediction
            target (str): The target

        Returns:
            Dict of the calculated metrics
        """
        return self([prediction], [target])

    def get_oracle_best_pred(self, predictions, target) -> str:
        best_pred_idx = np.argmax([
            self.calculate_single_example(p, target)[self.main_metric_key]
            for p in predictions
        ])
        return predictions[best_pred_idx]

    def oracle(self, predictions: List[List[str]], targets: List[str]) -> Dict:
        """
        Calculate the oracle score of a list of lists of predictions and a list
        of targets.

        Args:
            predictions (List[List[str]]): The nested list of predictions
            targets (List[str]): The list of targets

        Returns:
            The oracle metrics. NOTE, the keys from these metrics are the exact
            same as if it was not oracle.
        """
        best_predictions = []
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must be the same length.")
        for pred_list, target in zip(predictions, targets):
            best_predictions.append(self.get_oracle_best_pred(pred_list, target))
        return self(best_predictions, targets)


@Metric.register("exact-match")
class ExactMatch(Metric):

    def __init__(self):
        super(ExactMatch, self).__init__('exact-match', main_metric_key='em')

    def __call__(self, predictions: List[str], targets: List[str]) -> Dict:
        return {
            "em": sum(p == t for p, t in zip(predictions, targets)) / len(targets) * 100
        }


@Metric.register("bleu")
class BLEU(Metric):

    def __init__(self):
        super(BLEU, self).__init__('bleu', main_metric_key='bleu')
        self._bleu_kwargs = dict(
            smooth_method="exp",
            smooth_value=0.0,
            lowercase=False,
            tokenize="intl",
        )

    def __call__(self, predictions: List[str], targets: List[str]) -> Dict:
        # This came from the t5 repo
        targets = prepare_references_for_bleu(targets)

        bleu_score = sacrebleu.corpus_bleu(
            predictions,
            targets,
            force=False,
            use_effective_order=False,
            **self._bleu_kwargs
        )
        return {"bleu": bleu_score.score}

    def calculate_single_example(self, prediction: str, target: str) -> Dict:
        result = sacrebleu.sentence_bleu(
            prediction,
            [target],
            use_effective_order=True,
            **self._bleu_kwargs
        )
        return {'bleu': result.score}
