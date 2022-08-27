from typing import Dict, List
from collections import defaultdict

from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedStringSpan


KINDS = {
    'address': (
        "prov", "city", "district", "devzone", "town", "community", "village_group",
        "road", "roadno", "poi", "subpoi", "houseno", "cellno", "floorno", "roomno",
        "assist", "distance", "intersection", "others", "detail", "redundant"
    ),
    'ecommerce': ("HP", "HC")
}

# KIND_TO_UNUSED: Dict[str, str] = {k: f"[unused{i}]" for i, k in enumerate(KINDS, 1)}

def label_id_map(name: str):
    kinds = KINDS[name]
    id_to_tag: Dict[int, str] = {0: "O"}
    for k in kinds:
        for p in "BIES":
            tag = p + "-" + k
            index = len(id_to_tag)
            id_to_tag[index] = tag
    tag_to_id: Dict[str, int] = {v: k for k, v in id_to_tag.items()}
    return id_to_tag, tag_to_id


def iobes_to_spans(tags: List[str], _=None) -> List[TypedStringSpan]:
    spans = list()
    label, start, end = None, None, None
    for i, tag in enumerate(tags):
        if tag[0] in "OBS":
            assert start is None
        elif tag[0] in "IE":
            assert start is not None
        else:
            raise ValueError("wrong tag: ", tag)

        if tag.startswith("B-"):
            label, start, end = tag.split('-')[1], i, i
        elif tag.startswith("I-") or tag.startswith("E-"):
            assert tag.split('-')[1] == label
            end = i
            if tag.startswith("E-"):
                spans.append((label, (start, end)))
                start, end, label = None, None, None
        elif tag.startswith("S-"):
            spans.append((tag.split('-')[1], (i, i)))
    else:
        assert label is None
        assert start is None
        assert end is None

    return spans


def spans_to_iobes(spans: List[TypedStringSpan], length: int) -> List[str]:
    tags = ["O"] * length
    for label, (start, end) in spans:
        if start == end:
            tags[start] = "S-" + label
            continue
        tags[start] = "B-" + label
        for i in range(start + 1, end):
            tags[i] = "I-" + label
        tags[end] = "E-" + label
    return tags


VOTE_LABEL = (
    "city", "district", "town", "community", 'road', 'poi',
    "HC", "HP", "HCCX", "HPPX"
)


def _vote_span_label(counter: Dict[str, int]) -> str:
    return max(counter.items(), key=lambda x: x[1])[0]


def span_vote(
    texts: List[str],
    tags_array: List[List[TypedStringSpan]],
    suffix: bool = False,
) -> List[TypedStringSpan]:
    """
    texts: 所有文本，原始句子第一个，剩下是其余的检索
    tags_array: 与上面对应顺序的 iobes 表谦序列
    suffix: True 保留后缀不操作, False: 去常见后缀
    """
    if len(texts) < 2:
        return tags_array[0]

    if suffix:
        def remove_suffix(s):
            return s
    else:
        def remove_suffix(string: str):
            for suffix in ("省", "市", "区", "县", "乡", "镇", "街道", "社区"):
                if string.endswith(suffix):
                    return string[:-len(suffix)]
            return string

    counter = defaultdict(lambda: defaultdict(int))
    for text, tags in zip(texts, tags_array):
        spans = iobes_to_spans(tags)
        for label, (i, j) in spans:
            key = remove_suffix(text[i: j + 1])
            counter[key][label] += 1
    else:
        text, prediction = texts[0], tags_array[0]
        spans = iobes_to_spans(prediction)

    counter = {k: _vote_span_label(v) for k, v in counter.items()}
    voted = {(i, j): counter[remove_suffix(text[i: j + 1])] for _, (i, j) in spans}
    spans = {ij: l for l, ij in spans}
    for s, l in voted.items():
        if l in VOTE_LABEL:
            spans[s] = l
    spans = [(l, ij) for ij, l in spans.items()]
    # return spans_to_iobes(spans, len(text))
    return spans
