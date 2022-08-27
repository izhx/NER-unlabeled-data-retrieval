from typing import DefaultDict, Dict, List, Set
from collections import defaultdict
import logging

import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedStringSpan

logger = logging.getLogger(__name__)


@Metric.register("span_f1_dist")
class SpanF1Measure(Metric):
    """    """

    def __init__(self) -> None:
        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)
        self.training_finished = False

    def __call__(
        self,
        predictions: List[List[TypedStringSpan]],
        gold_labels: List[List[TypedStringSpan]]
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A batch of span list
        gold_labels : `torch.Tensor`, required.
            A batch of span list
        """
        for predicted_spans, gold_spans in zip(predictions, gold_labels):
            gold_spans = set(gold_spans)
            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset: bool = False):
        """
        # Returns

        `Dict[str, float]`
            A Dict per label containing following the span based metrics:
            - precision : `float`
            - recall : `float`
            - f1-measure : `float`

            Additionally, an `overall` key is included, which provides the precision,
            recall and f1-measure for all spans.
        """
        if reset and is_distributed() and not self.training_finished:
            # gather counters only when reseting can speed up training.
            def _gather(d: Dict) -> DefaultDict:
                dist.all_gather_object(dict_list, d)
                combined = defaultdict(int)
                for d in dict_list:
                    for k, v in d.items():
                        combined[k] += v
                return combined

            dict_list = [dict() for _ in range(dist.get_world_size())]
            self._true_positives = _gather(self._true_positives)
            self._false_positives = _gather(self._false_positives)
            self._false_negatives = _gather(self._false_negatives)

        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        pr = {"precision": list(), "recall": list()}
        for name, score in all_metrics.items():
            for k, v in pr.items():
                if k in name:
                    v.append(score)
                    break
        else:
            p, r = [sum(pr[k]) / len(pr[k]) if len(pr[k]) > 0 else 0 for k in ("precision", "recall")]
            all_metrics["precision-overall-MACRO"] = p
            all_metrics["recall-overall-MACRO"] = r
            all_metrics["f1-measure-overall-MACRO"] = 2.0 * (p * r) / (p + r + 1e-13)

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
