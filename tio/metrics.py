from typing import Dict, List
import sacrebleu

from tio.registrable import Registrable

from tio.common import prepare_references_for_bleu


class Metric(Registrable):
    def __call__(self, predictions: List[str], targets: List[str]) -> Dict:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, name, metric_dict):
        if name not in Registrable._registered_components[cls]:
            raise KeyError(f"{name} is not a valid metric")
        metric_cls = Registrable._registered_components[cls][name]
        return metric_cls(**metric_dict)


@Metric.register("exact-match")
class ExactMatch(Metric):
    def __call__(self, predictions: List[str], targets: List[str]) -> Dict:
        return {
            "em": sum(p == t for p, t in zip(predictions, targets)) / len(targets) * 100
        }


@Metric.register("bleu")
class BLEU(Metric):
    def __call__(self, predictions: List[str], targets: List[str]) -> Dict:
        # This came from the t5 repo
        targets = prepare_references_for_bleu(targets)

        bleu_score = sacrebleu.corpus_bleu(
            predictions,
            targets,
            smooth_method="exp",
            smooth_value=0.0,
            force=False,
            lowercase=False,
            tokenize="intl",
            use_effective_order=False,
        )
        return {"bleu": bleu_score.score}
