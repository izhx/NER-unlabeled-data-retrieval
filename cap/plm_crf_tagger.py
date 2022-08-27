from typing import Dict, List, Tuple, Any, cast

import torch
import torch.nn as nn

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import ConditionalRandomField
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator

from cap.data.utils import iobes_to_spans
from cap.embedding.plm_embedding import PretrainedLanguageModelEmbedding
from cap.training.metrics.span_f1_measure import SpanF1Measure


@Model.register("plm_crf_tagger")
class PlmCrfTagger(Model):
    """
    The `CrfTagger` encodes a sequence of text with a `Seq2SeqEncoder`,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.
    Registered as a `Model` with name "crf_tagger".
    # Parameters
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    model_name_or_path : `str`, required
        Pretrained model.
    encoder : `Seq2SeqEncoder`
        The encoder that we will use in between embedding tokens and predicting output tags.
    include_start_end_transitions : `bool`, optional (default=`True`)
        Whether to include start and end transition parameters in the CRF.
    dropout:  `float`, optional (default=`None`)
        Dropout probability.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    top_k : `int`, optional (default=`1`)
        If provided, the number of parses to return from the crf in output_dict['top_k_tags'].
        Top k parses are returned as a list of dicts, where each dictionary is of the form:
        {"tags": List, "score": float}.
        The "tags" value for the first dict in the list for each data_item will be the top
        choice, and will equal the corresponding item in output_dict['tags']
    """

    def __init__(
        self,
        vocab: Vocabulary,
        plm: Dict[str, Any],
        encoder: Seq2SeqEncoder,
        embedding_dropout: float = 0.5,
        encoder_dropout: float = 0.2,
        top_k: int = 1,
        label_namespace: str = "labels",
        include_start_end_transitions: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.plm_embedder = PretrainedLanguageModelEmbedding(**plm)
        self.encoder = encoder
        self.embedding_dropout = nn.Dropout(embedding_dropout) if embedding_dropout > 0 else nn.Identity()
        self.encoder_dropout = nn.Dropout(encoder_dropout) if encoder_dropout > 0 else nn.Identity()

        self.id_to_tag = vocab.get_index_to_token_vocabulary(label_namespace)
        self.tag_projection_layer = nn.Linear(self.encoder.get_output_dim(), len(self.id_to_tag))

        constraints = allowed_transitions(self.id_to_tag)
        self.crf = ConditionalRandomField(
            len(self.id_to_tag), constraints, include_start_end_transitions=include_start_end_transitions
        )

        self.top_k = top_k
        self._f1_metric = SpanF1Measure()

        check_dimensions_match(
            self.plm_embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )
        initializer(self)

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        mask: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        **kwargs,  # to allow for a more general dataset reader that passes args we don't need
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters
        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, num_tokens)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels of shape
            `(batch_size, num_tokens)`.
        metadata : `List[Dict[str, Any]]`, optional, (default = `None`)
            metadata containg the original words in the sentence to be tagged under a 'words' key.
        # Returns
        An output dictionary consisting of:
        logits : `torch.FloatTensor`
            The logits that are the output of the `tag_projection_layer`
        mask : `torch.BoolTensor`
            The text field mask for the input tokens
        tags : `List[List[int]]`
            The predicted tags using the Viterbi algorithm.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised. Only computed if gold label `tags` are provided.
        """
        embedded_text = self.plm_embedder(**tokens['bert'])
        embedded_text = self.embedding_dropout(embedded_text)
        embedded_text = embedded_text[:, 1: mask.size(1) + 1]
        output_dict = self.encode_decode_loss(embedded_text, mask, metadata, **kwargs)
        return output_dict

    def encode_decode_loss(
        self,
        embedded_text: torch.Tensor,
        mask: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        tags: torch.LongTensor = None,
        **kwargs,  # to allow for a more general dataset reader that passes args we don't need
    ) -> Dict[str, torch.Tensor]:
        """ encode decode loss """
        if hasattr(self, 'encoder') and not isinstance(self.encoder, PassThroughEncoder):
            encoded_text = self.encoder(embedded_text, mask)
            encoded_text = self.encoder_dropout(encoded_text)
        else:
            encoded_text = embedded_text

        logits = self.tag_projection_layer(encoded_text)
        best_paths = self.crf.viterbi_tags(logits, mask, top_k=self.top_k)

        # Just get the top tags and ignore the scores.
        predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])
        predicted_tags = [self.decode_tags(t) for t in predicted_tags]
        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if self.top_k > 1:
            output["top_k_tags"] = best_paths

        if tags is not None:
            tags = tags[:, 1: mask.size(1) + 1]
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, tags, mask)
            output["loss"] = -log_likelihood

            # feed into the metrics
            gold_spans = [m['gold_spans'] for m in metadata]
            predictions = [iobes_to_spans(i) for i in predicted_tags]
            self._f1_metric(predictions, gold_spans)

        if metadata is not None:
            output["text"] = [x["text"] for x in metadata]
            if tags is not None:
                output["gold_tags"] = [x["gold_tags"] for x in metadata]
        return output

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        `output_dict["tags"]` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """

        def decode_top_k_tags(top_k_tags):
            return [
                {"tags": self.decode_tags(scored_path[0]), "score": scored_path[1]}
                for scored_path in top_k_tags
            ]

        if "top_k_tags" in output_dict:
            output_dict["top_k_tags"] = [decode_top_k_tags(t) for t in output_dict["top_k_tags"]]

        return output_dict

    def decode_tags(self, tags: List[int]) -> List[str]:
        return [self.id_to_tag[tag] for tag in tags]

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_dict = self._f1_metric.get_metric(reset=reset)
        if reset:
            return f1_dict
        else:
            key = "f1-measure-overall"
            return {key: f1_dict[key]}

    default_predictor = "conll"


def allowed_transitions(labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    # Parameters

    labels : `Dict[int, str]`, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    # Returns

    `List[Tuple[int, int]]`
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed_iobes(from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed_iobes(
    from_tag: str, from_entity: str, to_tag: str, to_entity: str
):
    """
    Given a constraint type and strings `from_tag` and `to_tag` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    # Parameters

    from_tag : `str`, required
        The tag that the transition originates from. For example, if the
        label is `I-PER`, the `from_tag` is `I`.
    from_entity : `str`, required
        The entity corresponding to the `from_tag`. For example, if the
        label is `I-PER`, the `from_entity` is `PER`.
    to_tag : `str`, required
        The tag that the transition leads to. For example, if the
        label is `I-PER`, the `to_tag` is `I`.
    to_entity : `str`, required
        The entity corresponding to the `to_tag`. For example, if the
        label is `I-PER`, the `to_entity` is `PER`.

    # Returns

    `bool`
        Whether the transition is allowed under the given `constraint_type`.
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False
    if from_tag == "START":
        return to_tag in ("B", "S", "O")
    if to_tag == "END":
        return from_tag in ("E", "S", "O")
    return any(
        [
            # Can only transition to B, S or O from E, S or O.
            to_tag in ("B", "S", "O") and from_tag in ("E", "S", "O"),
            # Can only transition to I-x from B-x, where
            # x is the same tag.
            to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            # Can only transition to E-x from B-x or I-x, where
            # x is the same tag.
            to_tag == "E" and from_tag in ("B", "I") and from_entity == to_entity,
        ]
    )
