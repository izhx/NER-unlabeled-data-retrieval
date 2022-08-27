"""
The wrapper script.
"""

import argparse
from main import predict
import os
import json
import time
from typing import Dict, List, Iterable


_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--cuda', '-c', type=str, default='', help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--name', '-n', type=str, default='eco-ret', help='save name.')
_ARG_PARSER.add_argument('--seed', type=str, default='42', help='random seed.')
_ARG_PARSER.add_argument('--size', type=int, default=100, help='hits size.')
_ARG_PARSER.add_argument('--test', '-t', default=True, action="store_true", help='test model')

_ARGS = _ARG_PARSER.parse_args()

os.environ["OMP_NUM_THREADS"] = '8'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda

PREDICTED_FILE_NAME = "test_prediction.json"


def printf(*args):
    print(time.asctime(), "-", os.getpid(), ":", *args)


def jsonline_iter(path) -> Iterable[Dict]:
    with open(path) as file:
        for line in file:
            obj = json.loads(line)
            if obj:
                yield obj


def dump_jsonline(path: str, data: List):
    with open(path, mode='w') as file:
        for r in data:
            file.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("dump ", path)


def vote(predicted_file: str, size: int):
    import json
    from cap.data.utils import iobes_to_spans, spans_to_iobes, span_vote
    from cap.training.metrics.span_f1_measure import SpanF1Measure
    from conlleval import evaluate_conll_file

    metric = SpanF1Measure()
    lines = list()
    with open(predicted_file) as file:
        for row in file:
            ins = json.loads(row)
            texts = [ins['text']] + ins['hits'][:size]
            tags_array = [ins['predicted_tags']] + ins['predicted_tags_for_hits'][:size]
            spans = span_vote(texts, tags_array, suffix=True)
            tags = spans_to_iobes(spans, len(ins['text']))
            for items in zip(ins['text'], ins['tags'], tags):
                items = ('-', *items[1:])
                lines.append('\t'.join(items) + '\n')
            lines.append('\n')
            metric([iobes_to_spans(tags)], [iobes_to_spans(ins['tags'])])
            # evaluate_conll_file(lines)
            # print(sum(metric._false_negatives.values()) + sum(metric._true_positives.values()))
            # lines = list()
            # metric.get_metric(reset=True)

    metrics = metric.get_metric()
    print(json.dumps(metrics, indent=2))
    with open(predicted_file.replace("test_prediction", "metric_vote"), mode='w') as file:
        json.dump(metrics, file, indent=2)
        print('write metrics to', file.name)
    evaluate_conll_file(lines)


def predict(
    serialization_dir: str,
    seed: int = 42,
    cuda: str = '',
    test=False
):
    import torch
    from allennlp.common import Params, util
    from allennlp.data import DatasetReader
    from allennlp.models import Model
    from cap import RetrievalDatasetReader

    def predict(input_file: str):
        from allennlp.data.batch import Batch
        from allennlp.nn import util

        def to_instance(**kwargs):
            ins = dataset_reader.text_to_instance(**kwargs)
            dataset_reader.apply_token_indexers(ins)
            return ins

        def predict_batch(instances):
            dataset = Batch(instances)
            dataset.index_instances(model.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = model.make_output_human_readable(model(**model_input))
            return outputs['tags']

        if is_ret:
            def predict_one(obj, batch_size=52):
                instances = [to_instance(text=obj['text'], hits=obj['hits'])]
                instances.extend([
                    to_instance(text=h, hits=obj['hits_for_hits'][i]) 
                    for i, h in enumerate(obj['hits'][:100])
                ])
                outs = list()
                for i in range(0, len(instances), batch_size):
                    outs.extend(predict_batch(instances[i: i + batch_size]))
                return outs
        else:
            def predict_one(obj: Dict):
                instances = [to_instance(text=obj['text'])]
                instances.extend([to_instance(text=h) for h in obj['hits'][:100]])
                return predict_batch(instances)

        raw_data = [i for i in jsonline_iter(input_file)]
        printf("Read", len(raw_data), "instances from:", input_file)
        cuda_device = model._get_prediction_device()

        for i, obj in enumerate(raw_data):
            if i % 100 == 0:
                printf(i, '/', len(raw_data))
            pred = predict_one(obj)
            obj['predicted_tags'], obj['predicted_tags_for_hits'] = pred[0], pred[1:]

        dump_jsonline(f"{serialization_dir}/{PREDICTED_FILE_NAME}", raw_data)
        return

    cuda_device = -1 if _ARGS.cuda == '' else 0
    params = Params.from_file(serialization_dir + "/config.json")
    util.prepare_environment(params)
    dataset_reader = DatasetReader.from_params(params['dataset_reader'])
    printf("loding model from", serialization_dir)
    model = Model.load(params, serialization_dir, cuda_device=cuda_device)
    is_ret = isinstance(dataset_reader, RetrievalDatasetReader)

    if test:
        from allennlp.data import DataLoader
        from allennlp.training.util import evaluate as allen_evaluate

        data_loader = DataLoader.from_params(
            params['data_loader'], reader=dataset_reader, data_path=params['test_data_path']
        )
        data_loader.index_with(model.vocab)
        test_metrics = allen_evaluate(model, data_loader, cuda_device)
        string = json.dumps(test_metrics, indent=2)
        printf(string)

    with torch.no_grad():
        model.eval()
        if params['test_data_path'].endswith('.json'):
            suffix = '.ret'
        elif params['test_data_path'].endswith('.ret'):
            suffix = '.h4h'
        else:
            raise Exception
        # suffix = '.ret' if params['test_data_path'].endswith('.json') else ''
        predict(params['test_data_path'] + suffix)


if __name__ == "__main__":
    serialization_dir = f"results/{_ARGS.name}-{_ARGS.seed}"
    pred_file = f"{serialization_dir}/{PREDICTED_FILE_NAME}"
    if not os.path.isfile(pred_file):
        predict(serialization_dir, _ARGS.seed, _ARGS.cuda, _ARGS.test)
    vote(pred_file, _ARGS.size)
