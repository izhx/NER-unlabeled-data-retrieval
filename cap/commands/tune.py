"""
Modified from https://github.com/himkt/allennlp-optuna/blob/main/allennlp_optuna/commands/tune.py
"""

from typing import List, Optional, Union
import argparse
import json
import logging
import os
import shutil

from allennlp.commands.subcommand import Subcommand
from allennlp.common.params import evaluate_file, _environment_variables
import optuna
from optuna import Trial
from optuna.integration import AllenNLPExecutor
from overrides import overrides

logger = logging.getLogger(__name__)


def tune_from_args(args: argparse.Namespace) -> str:
    return tune(
        param_path=args.param_path,
        optuna_path=args.optuna_path,
        serialization_dir=args.serialization_dir,
        study_name=args.study_name,
        storage=args.storage,
        metrics=args.metrics,
        timeout=args.timeout,
        n_trials=args.n_trials,
        direction=args.direction,
        load_if_exists=args.load_if_exists,
        include_package=args.include_package,
        skip_exception=args.skip_exception
    )


def tune(
    param_path: str,
    optuna_path: str,
    serialization_dir: str,
    study_name: str,
    storage: str,
    metrics: str = "best_validation_loss",
    timeout: float = None,
    n_trials: int = 50,
    direction: str = "maximize",
    load_if_exists: bool = False,
    include_package: Optional[Union[str, List[str]]] = None,
    skip_exception: bool = False
) -> str:
    os.makedirs(serialization_dir, exist_ok=True)

    if optuna_path is not None and os.path.isfile(optuna_path):
        optuna_config = json.loads(evaluate_file(
            optuna_path, ext_vars=_environment_variables())
        )
    else:
        raise RuntimeError(f"<{optuna_path}> is not a optuna config file!")

    def objective(trial: Trial) -> float:
        for hparam in optuna_config["hparams"]:
            attr_type = hparam["type"]
            suggest = getattr(trial, "suggest_{}".format(attr_type))
            suggest(**hparam["attributes"])

        trial_serialization_dir = os.path.join(
            serialization_dir, "trial_{}".format(trial.number)
        )
        executor = AllenNLPExecutor(
            trial=trial,  # trial object
            config_file=param_path,  # path to jsonnet
            serialization_dir=trial_serialization_dir,
            metrics=metrics,
            include_package=include_package,
            # force=True,
            file_friendly_logging=True
        )
        return executor.run()

    if "pruner" in optuna_config:
        pruner_class = getattr(optuna.pruners, optuna_config["pruner"]["type"])
        pruner = pruner_class(**optuna_config["pruner"].get("attributes", {}))
    else:
        pruner = None

    if "sampler" in optuna_config:
        sampler_class = getattr(optuna.samplers, optuna_config["sampler"]["type"])
        sampler = sampler_class(optuna_config["sampler"].get("attributes", {}))
    else:
        sampler = None

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=load_if_exists,
    )
    study.optimize(
        objective,
        n_trials=n_trials,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
        catch=(Exception,) if skip_exception else (),
    )

    print("\n\nNumber of finished trials: ", len(study.trials))
    trial = study.best_trial
    print("Best trial:", trial.number)
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_trial_dir = f"{serialization_dir}/trial_{trial.number}"
    best_archive_dir = serialization_dir + "_best"
    shutil.copytree(best_trial_dir, best_archive_dir)
    shutil.rmtree(serialization_dir, ignore_errors=True)
    print(f"\nSaved best AllenNLP archives to `{best_archive_dir}`.")
    return best_archive_dir


@Subcommand.register("tune")
class Tune(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(self.name, description=description, help="Optimize hyperparameter of a model.")

        subparser.add_argument(
            "param_path",
            type=str,
            help="path to parameter file describing the model to be trained",
        )

        subparser.add_argument(
            "optuna_path",
            type=str,
            help="path to optuna config file",
            default="hyper_params.json",
        )

        subparser.add_argument(
            "--include-package",
            type=Union[str, List[str]],
            help="allennlp packages to include",
            default=None
        )

        subparser.add_argument(
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        # ---- Optuna -----

        subparser.add_argument(
            "--load-if-exists",
            default=False,
            action="store_true",
            help="If specified, the creation of the study is skipped "
            "without any error when the study name is duplicated.",
        )

        subparser.add_argument(
            "--direction",
            type=str,
            choices=("minimize", "maximize"),
            default="minimize",
            help="Set direction of optimization to a new study. Set 'minimize' "
            "for minimization and 'maximize' for maximization.",
        )

        subparser.add_argument(
            "--n-trials",
            type=int,
            help="The number of trials. If this argument is not given, as many " "trials run as possible.",
            default=50,
        )

        subparser.add_argument(
            "--timeout",
            type=float,
            help="Stop study after the given number of second(s). If this argument"
            " is not given, as many trials run as possible.",
        )

        subparser.add_argument(
            "--study-name", default=None, help="The name of the study to start optimization on."
        )

        subparser.add_argument(
            "--storage",
            type=str,
            help=(
                "The path to storage. "
                "allennlp-optuna supports a valid URL" "for sqlite3, mysql, postgresql, or redis."
            ),
            default="sqlite:///allennlp_optuna.db",
        )

        subparser.add_argument(
            "--metrics",
            type=str,
            help="The metrics you want to optimize.",
            default="best_validation_loss",
        )

        subparser.add_argument(
            "--skip-exception",
            action="store_true",
            help=(
                "If this option is specified, optimization won't stop even when it catches an exception."
                " Note that this option is experimental and it could be changed or removed in future development."
            ),
        )

        subparser.set_defaults(func=tune_from_args)
        return subparser
