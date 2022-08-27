"""
The wrapper script.
"""

import argparse
import shutil
import os
import sys

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--config', type=str, default='add', help='configuration file name.')
_ARG_PARSER.add_argument('--cuda', '-c', type=str, default='', help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--name', '-n', type=str, default='debug', help='save name.')
_ARG_PARSER.add_argument('--predict', '-p', default=False, action="store_true")

_ARG_PARSER.add_argument(
    "--node-rank", type=int, default=0, help="rank of this node in the distributed setup"
)
_ARG_PARSER.add_argument(
    "--file-friendly-logging",
    action="store_true",
    default=False,
    help="outputs tqdm status on separate lines and slows tqdm refresh rate",
)
_ARG_PARSER.add_argument(
    "-f",
    "--force",
    action="store_true",
    # default=True,
    required=False,
    help="overwrite the output directory if it exists",
)

_ARG_PARSER.add_argument('--optuna', '-o', default=False, action="store_true",
                         help="search hyper-parameters?")
_ARG_PARSER.add_argument('--hparams', type=str, default='bc_optuna', help='hparams file name.')
_ARG_PARSER.add_argument('--storage', type=str, default='mysql://opt:pass@localhost:3336/optuna',
                         help='storage file name.')
_ARG_PARSER.add_argument('--trials', type=int, default=128, help='number of trials to train a model')
_ARG_PARSER.add_argument('--timeout', type=int, default=12, help='threshold for executing time (hour)')
_ARG_PARSER.add_argument('--is-worker', default=False, action="store_true", help='parallel optimization.')

_ARG_PARSER.add_argument('--bert-name', type=str, default='nezha-base',
                         help="如果配置文件是 nbc_allen 这种把 model_name 写死的，此参数无用")
_ARG_PARSER.add_argument('--seed', type=str, default='42', help='random seed.')
_ARG_PARSER.add_argument('--batch-size', type=str, default='16', help='batch size')

_ARG_PARSER.add_argument('--pseudo', type=str, default='', help='tri-training file')
_ARG_PARSER.add_argument('--part', type=str, default='', help='tri-training part')
_ARG_PARSER.add_argument('--low', type=str, default='', help='low resource name')


_ARGS = _ARG_PARSER.parse_args()

os.environ["OMP_NUM_THREADS"] = '8'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda
os.environ['BERT_MODEL_NAME'] = _ARGS.bert_name

os.environ['BATCH_SIZE'] = _ARGS.batch_size
os.environ['RANDOM_SEED'] = _ARGS.seed

os.environ['TT_PSEUDO'] = _ARGS.pseudo
os.environ['TT_PART'] = _ARGS.part
os.environ['DATA_LOW'] = _ARGS.low


def evaluate(serialization_dir: str, params):
    import json
    from allennlp.data import DataLoader, DatasetReader
    from allennlp.models import Model
    from allennlp.training.util import evaluate as allen_evaluate

    cuda_device = -1 if _ARGS.cuda == '' else 0
    dataset_reader = DatasetReader.from_params(params['dataset_reader'])
    params['data_loader']['batch_sampler']['batch_size'] = 20
    data_loader = DataLoader.from_params(
        params['data_loader'], reader=dataset_reader, data_path=params['test_data_path']
    )
    model = Model.load(params, serialization_dir, cuda_device=cuda_device)
    data_loader.index_with(model.vocab)
    test_metrics = allen_evaluate(
        model, data_loader, cuda_device, output_file=serialization_dir + "/metric_test.json"
    )
    string = json.dumps(test_metrics, indent=2)
    print(string)


def predict(serialization_dir: str):
    from allennlp import commands
    from allennlp.common.params import Params
    from conlleval import evaluate_conll_file

    params = Params.from_file(serialization_dir + "/config.json")
    archive_file = serialization_dir + "/model.tar.gz"

    return evaluate(serialization_dir, params)

    if os.path.isfile(archive_file):
        pass
    elif os.path.isfile(serialization_dir + "/best.th"):
        from allennlp.models.archival import archive_model
        evaluate(serialization_dir, params)
        # return
        print("archiving the best model from", serialization_dir + "/best.th")
        archive_model(serialization_dir, "best.th")
    else:
        raise RuntimeError("can not predict without best model!")

    argv = [
        "allennlp",  # command name, not used by main
        "predict",
        archive_file,  # archive_file
        params["test_data_path"],
        "--output-file", serialization_dir + "/predict",  # 此文件可输入 conlleval
        "--silent",
        "--cuda-device=0",
        "--use-dataset-reader"
    ]
    print(" ".join(argv))
    sys.argv = argv
    commands.main()
    with open(serialization_dir + "/predict") as file:
        evaluate_conll_file(file)


def main(args: argparse.Namespace):
    from allennlp.common.params import Params
    import cap

    study_name = f"{args.name}-{args.seed}"
    serialization_dir = "results/" + study_name

    if args.predict:
        return predict(serialization_dir)

    config_file = f"config/{args.config}.jsonnet"
    params = Params.from_file(config_file)

    if args.optuna:
        import optuna
        from cap.commands.tune import tune

        if args.debug or args.force:
            try:
                optuna.delete_study(study_name, args.storage)
            except Exception as e:
                print(e)

        if "mysql" in args.storage:
            import pymysql
            pymysql.install_as_MySQLdb()

        tune(
            config_file,
            f"config/{args.hparams}.jsonnet",
            serialization_dir,
            study_name,
            args.storage,
            metrics="test_f1-measure-overall",
            timeout=args.timeout * 3600,
            n_trials=args.trials,
            direction="maximize",
            load_if_exists=args.is_worker
        )

    else:  # distributed hook
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

        from allennlp.commands.train import train_model

        # Training will fail if the serialization directory already
        # has stuff in it. If you are running the same training loop
        # over and over again for debugging purposes, it will.
        # Hence we wipe it out in advance.
        # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
        if os.path.isdir(serialization_dir) and args.force:
            shutil.rmtree(serialization_dir, ignore_errors=True)

        train_model(
            params,
            serialization_dir,
            node_rank=args.node_rank,
            file_friendly_logging=args.file_friendly_logging
        )

    if args.optuna:
        serialization_dir += "_best"
    # 最后输出预测文件, 但是用 conlleval 验证过了，和 allennlp 输出的分数一致
    # predict(serialization_dir)


if __name__ == "__main__":
    main(_ARGS)
