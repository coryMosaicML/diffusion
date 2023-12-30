# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate model."""

import operator
from typing import List

import hydra
from composer import Algorithm, ComposerModel
from composer.algorithms.low_precision_groupnorm import apply_low_precision_groupnorm
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from composer.core import Precision
from composer.loggers import LoggerDestination
from composer.utils import reproducibility
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.multimodal import CLIPScore

from diffusion.evaluation.clean_fid_eval import CleanFIDEvaluator
from diffusion.evaluation.pickscore_eval import PickScoreEvaluator


def evaluate(config: DictConfig) -> None:
    """Evaluate a model with the appropriate evaluator."""
    if 'CleanFIDEvaluator' in config.evaluator._target_:
        return clean_fid_evaluate(config)
    elif 'PickScoreEvaluator' in config.evaluator._target_:
        return pickscore_evaluate(config)
    else:
        raise ValueError(f'Unknown evaluator {config.evaluator._target_}')


def clean_fid_evaluate(config: DictConfig) -> None:
    """Evaluate a model usinf the clean fid evaluator.

    Args:
        config (DictConfig): Configuration composed by Hydra
    """
    reproducibility.seed_all(config.seed)

    # The model to evaluate
    model: ComposerModel = hydra.utils.instantiate(config.model)

    # The dataloader to use for evaluation
    eval_dataloader: DataLoader = hydra.utils.instantiate(config.eval_dataloader)

    # The CLIPScores metric to use for evaluation
    clip_metric: CLIPScore = hydra.utils.instantiate(config.clip_metric)

    # Build list of loggers and algorithms.
    logger: List[LoggerDestination] = []
    algorithms: List[Algorithm] = []

    # Set up logging for results
    if 'logger' in config:
        for log, lg_conf in config.logger.items():
            if '_target_' in lg_conf:
                print(f'Instantiating logger <{lg_conf._target_}>')
                if log == 'wandb':
                    container = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
                    # use _partial_ so it doesn't try to init everything
                    wandb_logger = hydra.utils.instantiate(lg_conf, _partial_=True)
                    logger.append(wandb_logger(init_kwargs={'config': container}))
                else:
                    logger.append(hydra.utils.instantiate(lg_conf))

    # Some algorithms should also be applied at inference time
    if 'algorithms' in config:
        for ag_name, ag_conf in config.algorithms.items():
            if '_target_' in ag_conf:
                print(f'Instantiating algorithm <{ag_conf._target_}>')
                algorithms.append(hydra.utils.instantiate(ag_conf))
            elif ag_name == 'low_precision_groupnorm':
                surgery_target = model
                if 'attribute' in ag_conf:
                    surgery_target = operator.attrgetter(ag_conf.attribute)(model)
                apply_low_precision_groupnorm(
                    model=surgery_target,
                    precision=Precision(ag_conf['precision']),
                    optimizers=None,
                )
            elif ag_name == 'low_precision_layernorm':
                surgery_target = model
                if 'attribute' in ag_conf:
                    surgery_target = operator.attrgetter(ag_conf.attribute)(model)
                apply_low_precision_layernorm(
                    model=surgery_target,
                    precision=Precision(ag_conf['precision']),
                    optimizers=None,
                )

    evaluator: CleanFIDEvaluator = hydra.utils.instantiate(
        config.evaluator,
        model=model,
        eval_dataloader=eval_dataloader,
        clip_metric=clip_metric,
        loggers=logger,
    )

    def evaluate_model():
        evaluator.evaluate()

    return evaluate_model()


def pickscore_evaluate(config: DictConfig) -> None:
    """Evaluate a pair of models using the pickscore evaluator.

    Args:
        config (DictConfig): Configuration composed by Hydra
    """
    reproducibility.seed_all(config.seed)

    # The baseline model
    baseline_model: ComposerModel = hydra.utils.instantiate(config.baseline_model)
    # The model to evaluate
    model: ComposerModel = hydra.utils.instantiate(config.model)

    # The dataloader to use for evaluation
    eval_dataloader: DataLoader = hydra.utils.instantiate(config.eval_dataloader)

    # Build list of loggers and algorithms.
    logger: List[LoggerDestination] = []

    # Set up logging for results
    if 'logger' in config:
        for log, lg_conf in config.logger.items():
            if '_target_' in lg_conf:
                print(f'Instantiating logger <{lg_conf._target_}>')
                if log == 'wandb':
                    container = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
                    # use _partial_ so it doesn't try to init everything
                    wandb_logger = hydra.utils.instantiate(lg_conf, _partial_=True)
                    logger.append(wandb_logger(init_kwargs={'config': container}))
                else:
                    logger.append(hydra.utils.instantiate(lg_conf))

    evaluator: PickScoreEvaluator = hydra.utils.instantiate(
        config.evaluator,
        baseline_model=baseline_model,
        model=model,
        eval_dataloader=eval_dataloader,
        loggers=logger,
    )

    def evaluate_model():
        evaluator.evaluate()

    return evaluate_model()
