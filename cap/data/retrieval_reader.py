import re
from typing import List, Optional
import logging

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data import Token, Instance

from cap.data.base_reader import BaseDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("retrieval")
class RetrievalDatasetReader(BaseDatasetReader):
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
        max_retrieval: int = 10,
        max_length: int = 250,
        merge_hits: bool = False,
        de_numbers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_retrieval = max_retrieval
        self.max_length = max_length
        self.merge_hits = merge_hits
        self.de_numbers = de_numbers

    def text_to_instance(  # type: ignore
        self,
        text: str,
        hits: List[str],
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Instance:
        """  """
        ins = super().text_to_instance(text, tags)
        tokens = ['[CLS]', *text, '[SEP]']

        texts = select_text(
            hits, text, self.max_retrieval, self.max_length, self.de_numbers
        )
        if self.merge_hits:
            texts = [merge_texts(texts)]

        for t in texts:
            tokens.extend(t)
            tokens.append("[SEP]")
        else:
            tokens = [t if t != '[SEP]' else '[unused1]' for t in tokens]
            tokens[-1] = '[SEP]'

        ins.fields['metadata'].metadata['hits'] = texts
        ins.fields["tokens"] = TextField([Token(w) for w in tokens])
        return ins


def select_text(
    hits: List[str],
    target: str,
    max_retrieval=12,
    max_length=250,
    de_numbers=True
) -> List[str]:
    selected, total_len = list(), len(target) + 2
    if de_numbers:
        target = re.sub(r'\d+', '0', ''.join(target.split()))
    for n, t in enumerate(hits):
        if de_numbers:
            t = re.sub(r'\d+', '0', ''.join(t.split()))
        if t in target:
            continue
        if len(selected) == 0:
            selected.append(t)
            total_len += len(t) + 1
            continue
        for i in range(len(selected)):
            s = selected[i]
            if t in s:
                break
            elif s in t:
                selected[i] = t
                total_len += len(t) - len(s)
                break
        else:
            # if string_similar(t, k) < thresh:
            if total_len + len(t) + 1 <= max_length and len(selected) < max_retrieval:
                selected.append(t)
                total_len += len(t) + 1
            else:
                break
    # assert len(selected) > 2
    # Counter[len(selected)] += 1
    return selected


SUB = re.compile(r"(.+)(?=.*\1)")

def merge_texts(texts: List[str]) -> str:
    subs = re.findall(r"(\S{1,})(?=.*\1)", ' '.join(texts))
    drop_ids = set(i for i, s in enumerate(subs) if len(s) < 2)
    for i, si in enumerate(subs):
        if i in drop_ids:
            continue
        for j, sj in enumerate(subs):
            if j in drop_ids:
                continue
            if j == i:
                continue
            if sj in si:
                drop_ids.add(j)
            elif si in sj:
                drop_ids.add(i)
                break
    remained = [s for i, s in enumerate(subs) if i not in drop_ids]
    return ','.join(remained)
