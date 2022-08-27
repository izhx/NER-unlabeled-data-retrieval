from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("conll")
class ConllPredictor(Predictor):
    """
    Predictor for the Address Parsing task.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"text": sentence})

    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        lines = list()
        if "gold_tags" in outputs:
            objs = [outputs["text"], outputs["gold_tags"], outputs["tags"]]
        else:
            objs = [outputs["text"], outputs["tags"]]
        for items in zip(*objs):
            lines.append("\t".join(items))
        return "\n".join(lines) + "\n"

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"text": "..."}`.
        Runs the underlying model, and adds the `"words"` to the output.
        """
        return self._dataset_reader.text_to_instance(**json_dict)
