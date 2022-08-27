"""
The wrapper script.
"""

import argparse
import os
import json
import time
from typing import Dict, List, Iterable


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
    printf("dump", len(data), "lines to", path)


def main(
    input_file_or_instances,
    name: str,
    seed: int = 42,
    batch_size: int = 48,
    cuda: str = '',
    test=True,
    prediction_name: str = "prediction"
):

    os.environ["OMP_NUM_THREADS"] = '8'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda

    import torch
    from allennlp.common import Params, util
    from allennlp.data import DatasetReader
    from allennlp.models import Model
    import cap

    def predict(raw_data: List[Dict], serialization_dir: str, prediction_name: str):
        from allennlp.data.batch import Batch
        from allennlp.nn import util
        from cap import RetrievalDatasetReader

        cuda_device = model._get_prediction_device()

        def predict_batch(batch: List):
            instances = list()
            for raw in batch:
                kwargs = {'text': raw['text']}
                if isinstance(dataset_reader, RetrievalDatasetReader):
                    kwargs['hits'] = raw['hits']
                ins = dataset_reader.text_to_instance(**kwargs)
                dataset_reader.apply_token_indexers(ins)
                instances.append(ins)

            dataset = Batch(instances)
            dataset.index_instances(model.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            # print(model_input['tokens']['bert']['token_ids'].size())
            outputs = model.make_output_human_readable(model(**model_input))

            key = 'predicted_tags' if 'tags' in batch[0] else 'tags'
            for i, obj in enumerate(batch):
                obj[key] = outputs['tags'][i]
            del dataset, model_input, outputs
            batch.clear()

        batch = list()
        for i, obj in enumerate(raw_data):
            if i % 1000 == 0:
                printf(i, '/', len(raw_data))
            batch.append(obj)
            if len(batch) >= batch_size:
                predict_batch(batch)
        else:
            if len(batch) > 0:
                predict_batch(batch)

        dump_jsonline(f"{serialization_dir}/{prediction_name}.json", raw_data)
        return


    cuda_device = -1 if _ARGS.cuda == '' else 0
    serialization_dir = f"results/{name}-{seed}"
    params = Params.from_file(serialization_dir + "/config.json")
    util.prepare_environment(params)
    dataset_reader = DatasetReader.from_params(params['dataset_reader'])
    printf("loding model from", serialization_dir)
    model = Model.load(params, serialization_dir, cuda_device=cuda_device)

    if test:
        from allennlp.data import DataLoader
        from allennlp.training.util import evaluate as allen_evaluate

        params['data_loader']['batch_sampler']['batch_size'] = batch_size
        data_loader = DataLoader.from_params(
            params['data_loader'], reader=dataset_reader, data_path=params['test_data_path']
        )
        data_loader.index_with(model.vocab)
        test_metrics = allen_evaluate(
            model, data_loader, cuda_device, output_file=serialization_dir + "/metric_test.json"
        )
        string = json.dumps(test_metrics, indent=2)
        printf(string)

    if isinstance(input_file_or_instances, str):
        raw_data = list(jsonline_iter(input_file_or_instances))
        printf("Read", len(raw_data), "instances from:", input_file_or_instances)
    elif isinstance(input_file_or_instances, list):
        raw_data = input_file_or_instances
        printf("Get", len(raw_data), "instances.")
    else:
        raise Exception

    with torch.no_grad():
        model.eval()
        predict(list(jsonline_iter(params['test_data_path'])), serialization_dir, "test")
        predict(raw_data, serialization_dir, prediction_name)


if __name__ == "__main__":
    _ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
    _ARG_PARSER.add_argument('--cuda', '-c', type=str, default='', help='gpu ids, like: 1,2,3')
    _ARG_PARSER.add_argument('--name', '-n', type=str, default='add-ret', help='save name.')
    _ARG_PARSER.add_argument('--seed', type=str, default='42', help='random seed.')
    _ARG_PARSER.add_argument('--batch-size', type=int, default=64, help='batch size')
    _ARG_PARSER.add_argument('--test', '-t', default=True, action="store_true", help='test model')
    _ARG_PARSER.add_argument(
        '--input-file', type=str, help='random seed.',
        default="/nas-alinlp/linzhang.zx/address/files/retrieval-pseudo/train-r0.json"
    )
    _ARGS = _ARG_PARSER.parse_args()

    main(
        _ARGS.input_file, _ARGS.name, _ARGS.seed, _ARGS.batch_size, _ARGS.cuda, _ARGS.test
    )
