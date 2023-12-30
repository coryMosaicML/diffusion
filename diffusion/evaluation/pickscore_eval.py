# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Runs pickscore evaluation for a pair of models."""

import os
from typing import List, Optional

import torch
import wandb
from composer import ComposerModel
from composer.core import get_precision_context
from composer.loggers import LoggerDestination, WandBLogger
from composer.utils import dist
from composer.utils.file_helpers import get_file
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from diffusion.evaluation.pickscore_metric import PickScoreMetric

# Local checkpoint params
LOCAL_BASELINE_CHECKPOINT_PATH = '/tmp/baseline_model.pt'
LOCAL_MODEL_CHECKPOINT_PATH = '/tmp/model.pt'


class PickScoreEvaluator:
    """Evaluator for the Pickscore of two models.

    Args:
        baseline_model (ComposerModel): The model to use as a baseline.
        model (ComposerModel): The model to compute the score of relative to the baseline.
        eval_dataloader (DataLoader): The dataloader to use for evaluation.
        baseline_load_path (str, optional): The path to load the baseline model from. Default: ``None``.
        model_load_path (str, optional): The path to load the model from. Default: ``None``.
        guidance_scales (List[float]): The guidance scales to use for evaluation.
            Default: ``[1.0]``.
        height (int): The height of the images to generate. Default: ``256``.
        width (int): The width of the images to generate. Default: ``256``.
        batch_size (int): The per-device batch size to use for evaluation. Default: ``16``.
        image_key (str): The key to use for the images in the dataloader. Default: ``'image'``.
        caption_key (str): The key to use for the captions in the dataloader. Default: ``'caption'``.
        load_strict_model_weights (bool): Whether or not to strict load model weights. Default: ``True``.
        loggers (List[LoggerDestination], optional): The loggers to use for logging results. Default: ``None``.
        seed (int): The seed to use for evaluation. Default: ``17``.
        output_dir (str): The directory to save results to. Default: ``/tmp/``.
        num_samples (int, optional): The maximum number of samples to generate. Depending on batch size, actual
            number may be slightly higher. If not specified, all the samples in the dataloader will be used.
            Default: ``None``.
        precision (str): The precision to use for evaluation. Default: ``'amp_fp16'``.
        prompts (List[str], optional): The prompts to use for image visualtization.
            Default: ``["A shiba inu wearing a blue sweater]``.

    """

    def __init__(self,
                 baseline_model: ComposerModel,
                 model: ComposerModel,
                 eval_dataloader: DataLoader,
                 baseline_load_path: Optional[str] = None,
                 model_load_path: Optional[str] = None,
                 guidance_scales: Optional[List[float]] = None,
                 height: int = 256,
                 width: int = 256,
                 batch_size: int = 16,
                 image_key: str = 'image',
                 caption_key: str = 'caption',
                 load_strict_model_weights: bool = True,
                 loggers: Optional[List[LoggerDestination]] = None,
                 seed: int = 17,
                 output_dir: str = '/tmp/',
                 num_samples: Optional[int] = None,
                 precision: str = 'amp_fp16',
                 prompts: Optional[List[str]] = None):
        self.baseline_model = baseline_model
        self.model = model
        self.tokenizer: PreTrainedTokenizerBase = baseline_model.tokenizer
        self.eval_dataloader = eval_dataloader
        self.baseline_load_path = baseline_load_path
        self.model_load_path = model_load_path
        self.guidance_scales = guidance_scales if guidance_scales is not None else [1.0]
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.image_key = image_key
        self.caption_key = caption_key
        self.loggers = loggers
        self.seed = seed
        self.output_dir = output_dir
        self.num_samples = num_samples if num_samples is not None else float('inf')
        self.precision = precision
        self.prompts = prompts if prompts is not None else ['A shiba inu wearing a blue sweater']
        self.sdxl = baseline_model.sdxl
        self.device = dist.get_local_rank()

        dist.initialize_dist('gpu')
        # Init loggers
        if self.loggers and dist.get_global_rank() == 0:
            for logger in self.loggers:
                if isinstance(logger, WandBLogger):
                    wandb.init(**logger._init_kwargs)

        # Load the baseline model
        if self.baseline_load_path is not None:
            if dist.get_local_rank() == 0:
                get_file(path=self.baseline_load_path, destination=LOCAL_BASELINE_CHECKPOINT_PATH)
            else:
                dist.local_rank_zero_download_and_wait(LOCAL_BASELINE_CHECKPOINT_PATH)
            state_dict = torch.load(LOCAL_BASELINE_CHECKPOINT_PATH)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            self.baseline_model.load_state_dict(state_dict['state']['model'], strict=load_strict_model_weights)
        self.baseline_model.to(self.device)

        # Load the model, leaving it on CPU for now.
        if self.model_load_path is not None:
            # SKIP DOWNLOADING IF THE FILE EXISTS FOR DEBUGGING
            if dist.get_local_rank() == 0 and not os.path.exists(LOCAL_MODEL_CHECKPOINT_PATH):
                get_file(path=self.model_load_path, destination=LOCAL_MODEL_CHECKPOINT_PATH)
            else:
                dist.local_rank_zero_download_and_wait(LOCAL_MODEL_CHECKPOINT_PATH)
            state_dict = torch.load(LOCAL_MODEL_CHECKPOINT_PATH, map_location='cpu')
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            self.model.load_state_dict(state_dict['state']['model'], strict=load_strict_model_weights)

        # Create the pickscore metric
        self.pickscore_metric = PickScoreMetric()
        # Move the metric to device
        if torch.cuda.is_available():
            self.pickscore_metric.cuda()

    def _generate_images(self, guidance_scale: float):
        """Core image generation function. Generates images at a given guidance scale.

        Args:
            guidance_scale (float): The guidance scale to use for image generation.
        """
        # Verify output dirs exist, if they don't, create them
        baseline_image_path = os.path.join(self.output_dir, f'baseline_images_gs_{guidance_scale}')
        if not os.path.exists(baseline_image_path) and dist.get_local_rank() == 0:
            os.makedirs(baseline_image_path)
        model_image_path = os.path.join(self.output_dir, f'model_images_gs_{guidance_scale}')
        if not os.path.exists(model_image_path) and dist.get_local_rank() == 0:
            os.makedirs(model_image_path)
        # Reset metric
        self.pickscore_metric.reset()
        # Storage for prompts
        prompts = {}
        # Iterate over the eval dataloader to generate the baseline images.
        num_batches = len(self.eval_dataloader)
        # Need to offset the seed by the number of batches * the rank
        starting_seed = self.seed + num_batches * dist.get_global_rank()
        for batch_id, batch in tqdm(enumerate(self.eval_dataloader)):
            # Break if enough samples have been generated
            if batch_id * self.batch_size * dist.get_world_size() >= self.num_samples:
                break
            captions = batch[self.caption_key]
            if captions.shape[1] > 1 and not self.baseline_model.sdxl:
                captions = captions[:, 0, :]
            elif captions.shape[1] == 1 and self.baseline_model.sdxl:
                raise ValueError('Cannot generate images with SDXL model from non-SDXL dataset.')
            # Ensure a new seed for each batch, as randomness in model.generate is fixed.
            seed = starting_seed + batch_id
            # Generate images from the captions
            with get_precision_context(self.precision):
                generated_images = self.baseline_model.generate(tokenized_prompts=captions,
                                                                height=self.height,
                                                                width=self.width,
                                                                guidance_scale=guidance_scale,
                                                                seed=seed,
                                                                crop_params=None,
                                                                input_size_params=None,
                                                                progress_bar=False)  # type: ignore
            # Get the prompts from the tokens
            if self.sdxl:
                # Decode with first tokenizer
                text_captions = self.tokenizer.tokenizer.batch_decode(captions[:, 0, :], skip_special_tokens=True)
            else:
                text_captions = self.tokenizer.batch_decode(captions, skip_special_tokens=True)
            # Save the baseline images
            for i, img in enumerate(generated_images):
                to_pil_image(img).save(f'{baseline_image_path}/{batch_id}_{i}_rank_{dist.get_global_rank()}.png')
                # Save the prompts and the seed.
                prompts[f'{batch_id}_{i}_rank_{dist.get_global_rank()}'] = (text_captions[i], seed)

        # Now run the model on the prompts, and update the pickscore metric along the way.
        # Move the baseline model to cpu, and the model to the device
        self.baseline_model = self.baseline_model.to('cpu')
        self.model = self.model.to(self.device)
        # Iterate over the prompts to generate comparison images
        for prompt_id, (prompt, seed) in tqdm(prompts.items()):
            # Generate images from the prompt
            with get_precision_context(self.precision):
                model_images = self.model.generate(prompt=prompt,
                                                   height=self.height,
                                                   width=self.width,
                                                   guidance_scale=guidance_scale,
                                                   seed=seed,
                                                   crop_params=None,
                                                   input_size_params=None,
                                                   progress_bar=False)  # type: ignore
            # Save the model images
            for i, img in enumerate(model_images):
                to_pil_image(img).save(f'{model_image_path}/{prompt_id}.png')
            # Load the baseline and model images
            baseline_image = Image.open(f'{baseline_image_path}/{prompt_id}.png')
            model_image = Image.open(f'{model_image_path}/{prompt_id}.png')
            # Update the pickscore metric
            self.pickscore_metric.update(prompts=[prompt], baseline_images=[baseline_image], model_images=[model_image])

    def _compute_metrics(self, guidance_scale: float):
        """Compute metrics for the generated images at a given guidance scale.

        Args:
            guidance_scale (float): The guidance scale to use for image generation.

        Returns:
            Dict[str, float]: The computed metrics.
        """
        metrics = {}
        # PickScore
        pickscore = self.pickscore_metric.compute()
        metrics['PickScore'] = pickscore
        print(f'{guidance_scale} PickScore: {pickscore}')
        return metrics

    def _generate_images_from_prompts(self, guidance_scale: float):
        """Generate images from prompts for visualization."""
        if self.prompts:
            with get_precision_context(self.precision):
                generated_images = self.model.generate(prompt=self.prompts,
                                                       height=self.height,
                                                       width=self.width,
                                                       guidance_scale=guidance_scale,
                                                       seed=self.seed)  # type: ignore
        else:
            generated_images = []
        return generated_images

    def evaluate(self):
        # Generate images and compute metrics for each guidance scale
        for guidance_scale in self.guidance_scales:
            dist.barrier()
            # Generate images and compute metrics
            self._generate_images(guidance_scale=guidance_scale)
            # Need to wait until all ranks have finished generating images before computing metrics
            dist.barrier()
            # Compute the metrics on the generated images
            metrics = self._compute_metrics(guidance_scale=guidance_scale)
            # Generate images from prompts for visualization
            generated_images = self._generate_images_from_prompts(guidance_scale=guidance_scale)
            # Log metrics and images on rank 0
            if self.loggers and dist.get_local_rank() == 0:
                for logger in self.loggers:
                    for metric, value in metrics.items():
                        logger.log_metrics({f'{guidance_scale}/{metric}': value})
                    for prompt, image in zip(self.prompts, generated_images):
                        logger.log_images(images=image, name=f'{prompt}_gs_{guidance_scale}')
