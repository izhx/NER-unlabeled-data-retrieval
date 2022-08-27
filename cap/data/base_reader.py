import json
from typing import Any, Dict, List, Iterable, Optional
import logging

import torch

from allennlp.data.dataset_readers.dataset_reader import DatasetReader, PathOrStr
from allennlp.data.fields import TextField, TensorField, Field, MetadataField, SequenceLabelField
from allennlp.data import Token, TokenIndexer, Instance

from cap.data.utils import iobes_to_spans  #, label_id_map

logger = logging.getLogger(__name__)


@DatasetReader.register("base")
class BaseDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    ``` { "text": "aaa", "tags": ["a", "a", "a"] } ```
    Registered as a `DatasetReader` with name "aep".
    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, required
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    label_namespace : `str`, optional (default=`labels`)
        Specifies the namespace for the chosen `tag_label`.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        extend: List[str] = None,
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._token_indexers = token_indexers
        self.label_namespace = label_namespace
        self.extend = extend
        # self.tag_to_id = label_id_map(dataset)[1]

    def _read(self, file_path: PathOrStr) -> Iterable[Instance]:
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            if 'test' in file_path:
                iterable = data_file  # final evalutation is not distributed
            else:
                iterable = self.shard_iterable(data_file)
            for line in iterable:
                if line.strip() != '':
                    obj = json.loads(line)
                    yield self.text_to_instance(**obj)
        if self.extend is not None and 'train' in file_path:
            for path in self.extend:
                with open(path, "r") as data_file:
                    logger.info("Reading instances from lines in file at: %s", path)
                    for line in self.shard_iterable(data_file):
                        if line.strip() != '':
                            obj = json.loads(line)
                            yield self.text_to_instance(**obj)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance["tokens"].token_indexers = self._token_indexers

    @staticmethod
    def basic_fields(text: str, tags: Optional[List[str]] = None) -> Dict[str, Field]:
        metadata: Dict[str, Any] = {"text": text}
        if tags is not None:
            spans = iobes_to_spans(tags)
            metadata["gold_tags"], metadata["gold_spans"] = tags, spans
        fields: Dict[str, Field] = {"metadata": MetadataField(metadata)}
        return fields

    def text_to_instance(  # type: ignore
        self,
        text: str,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Instance:
        """  """
        fields = self.basic_fields(text, tags)
        tokens = ['[CLS]', *text, '[SEP]']
        fields["tokens"] = TextField([Token(w) for w in tokens])
        fields["mask"] = TensorField(torch.ones(len(text)).long())
        if tags:
            # tag_ids = torch.tensor([self.tag_to_id[t] for t in tags]).long()
            # fields["tags"] = TensorField(tag_ids)
            tags = ['O'] + tags + ['O']
            fields["tags"] = SequenceLabelField(tags, fields["tokens"], self.label_namespace)
        return Instance(fields)
