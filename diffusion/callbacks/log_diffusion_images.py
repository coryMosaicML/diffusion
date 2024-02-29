# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Logger for generated images."""

from math import ceil
from typing import List, Optional, Tuple, Union

import torch
from composer import Callback, Logger, State
from composer.core import TimeUnit, get_precision_context
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel
from transformers import CLIPTextModel, CLIPTokenizer

from diffusion.models import ComposerDiffusionTransformer


class LogDiffusionImages(Callback):
    """Logs images generated from the evaluation prompts to a logger.

    Logs eval prompts and generated images to a table at
    the end of an evaluation batch.

    Args:
        prompts (List[str]): List of prompts to use for evaluation.
        size (int, Tuple[int, int]): Image size to use during generation.
            If using a tuple, specify as (height, width).  Default: ``256``.
        batch_size (int, optional): The batch size of the prompts passed to the generate function. If set to ``None``,
            batch_size is equal to the number of prompts. Default: ``1``.
        num_inference_steps (int): Number of inference steps to use during generation. Default: ``50``.
        guidance_scale (float): guidance_scale is defined as w of equation 2
            of the Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1.
            A larger guidance scale generates images that are more aligned to
            the text prompt, usually at the expense of lower image quality.
            Default: ``0.0``.
        rescaled_guidance (float, optional): Rescaled guidance scale. If not specified, rescaled guidance
            will not be used. Default: ``None``.
        seed (int, optional): Random seed to use for generation. Set a seed for reproducible generation.
            Default: ``1138``.
        use_table (bool): Whether to make a table of the images or not. Default: ``False``.
    """

    def __init__(self,
                 prompts: List[str],
                 size: Union[Tuple[int, int], int] = 256,
                 batch_size: Optional[int] = 1,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 0.0,
                 rescaled_guidance: Optional[float] = None,
                 seed: Optional[int] = 1138,
                 use_table: bool = False):
        self.prompts = prompts
        self.size = (size, size) if isinstance(size, int) else size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.rescaled_guidance = rescaled_guidance
        self.seed = seed
        self.use_table = use_table

        # Batch prompts
        batch_size = len(prompts) if batch_size is None else batch_size
        num_batches = ceil(len(prompts) / batch_size)
        self.batched_prompts = []
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            self.batched_prompts.append(prompts[start:end])

    def eval_start(self, state: State, logger: Logger):
        # Get the model object if it has been wrapped by DDP to access the image generation function.
        if isinstance(state.model, DistributedDataParallel):
            model = state.model.module
        else:
            model = state.model

        # Generate images
        with get_precision_context(state.precision):
            all_gen_images = []
            for batch in self.batched_prompts:
                gen_images = model.generate(
                    prompt=batch,  # type: ignore
                    height=self.size[0],
                    width=self.size[1],
                    guidance_scale=self.guidance_scale,
                    rescaled_guidance=self.rescaled_guidance,
                    progress_bar=False,
                    num_inference_steps=self.num_inference_steps,
                    seed=self.seed)
                all_gen_images.append(gen_images)
            gen_images = torch.cat(all_gen_images)

        # Log images to wandb
        for prompt, image in zip(self.prompts, gen_images):
            logger.log_images(images=image, name=prompt, step=state.timestamp.batch.value, use_table=self.use_table)


class LogAutoencoderImages(Callback):
    """Logs images from an autoencoder to compare real inputs to their autoencoded outputs.

    Args:
        image_key (str): Key in the batch to use for images. Default: ``'image'``.
        max_images (int): Maximum number of images to log. Default: ``10``.
        log_latents (bool): Whether to log the latents or not. Default: ``True``.
        use_table (bool): Whether to make a table of the images or not. Default: ``False``.
    """

    def __init__(self,
                 image_key: str = 'image',
                 max_images: int = 10,
                 log_latents: bool = True,
                 use_table: bool = False):
        self.image_key = image_key
        self.max_images = max_images
        self.log_latents = log_latents
        self.use_table = use_table

    def _scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Scale latents to be between 0 and 1 for visualization."""
        latents = latents - latents.min()
        latents = latents / latents.max()
        return latents

    def eval_batch_end(self, state: State, logger: Logger):
        # Only log once per eval epoch
        if state.eval_timestamp.get(TimeUnit.BATCH).value == 1:
            # Get the inputs
            images = state.batch[self.image_key]
            if self.max_images > images.shape[0]:
                max_images = images.shape[0]
            else:
                max_images = self.max_images

            # Get the model reconstruction
            outputs = state.model(state.batch)
            recon = outputs['x_recon']
            latents = outputs['latents']

            # Log images to wandb
            for i, image in enumerate(images[:max_images]):
                # Clamp the reconstructed image to be between 0 and 1
                recon_img = (recon[i] / 2 + 0.5).clamp(0, 1)
                logged_images = [image, recon_img]
                if self.log_latents:
                    logged_images += [self._scale_latents(latents[i][j]) for j in range(latents.shape[1])]
                logger.log_images(images=logged_images,
                                  name=f'Image (input, reconstruction, latents) {i}',
                                  step=state.timestamp.batch.value,
                                  use_table=self.use_table)


class LogTransformerImages(Callback):
    """Logs images generated from the evaluation prompts to a logger."""

    def __init__(self, latent: bool = False):
        self.latent = latent
        self.prompts = [
            'A photo of a shiba inu wearing a blue sweater', 'A majestic lion jumping from a big stone at night'
        ]
        self.height = 256
        self.width = 256
        self.patch_size = 16
        self.guidance_scale = 7.0
        self.num_timesteps = 50
        self.seed = 1138

        model_name = 'stabilityai/stable-diffusion-2-base'
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder').cpu()
        if self.latent:
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae').cpu()

    def _encode_prompt(self, prompt: str):
        """Encodes a prompt into a conditioning tensor."""
        tokenizer_out = self.tokenizer(prompt, padding='max_length', truncation=True, return_tensors='pt')
        input_ids, attention_mask = tokenizer_out['input_ids'], tokenizer_out['attention_mask']
        device = self.text_encoder.device
        text_encoding = self.text_encoder(input_ids.to(device), attention_mask=attention_mask.to(device))[0]
        text_coords = torch.arange(text_encoding.shape[0]).unsqueeze(0).unsqueeze(-1)
        return text_encoding, text_coords, attention_mask

    def _create_input_coords_and_mask(self, height: int, width: int):
        """Creates input coordinates for the input image."""
        num_H_patches = height // self.patch_size
        num_W_patches = width // self.patch_size
        input_coords = torch.tensor([(i, j) for i in range(num_H_patches) for j in range(num_W_patches)]).unsqueeze(0)
        input_mask = torch.ones(1, input_coords.shape[1])
        return input_coords, input_mask

    def _reconstruct_image_from_patches(self, patches, coords):
        C = patches.shape[1] // (self.patch_size * self.patch_size)
        # Calculate the height and width of the original image from the coordinates
        H = coords[:, 0].max() * self.patch_size + self.patch_size
        W = coords[:, 1].max() * self.patch_size + self.patch_size
        # Initialize an empty tensor for the reconstructed image
        img = torch.zeros((C, H, W))
        # Iterate over the patches and their coordinates
        for patch, (y, x) in zip(patches, self.patch_size * coords):
            # Reshape the patch to [C, patch_size, patch_size]
            patch = patch.view(C, self.patch_size, self.patch_size)
            # Place the patch in the corresponding location in the image
            img[:, y:y + self.patch_size, x:x + self.patch_size] = patch
        # Image is in the range [-1, 1], so shift and clamp it to [0, 1]
        img = (img / 2 + 0.5).clamp(0, 1)
        return img

    def eval_start(self, state: State, logger: Logger):
        # Get the model object if it has been wrapped by DDP to access the image generation function.
        if isinstance(state.model, DistributedDataParallel):
            model = state.model.module
        else:
            model = state.model
        assert isinstance(model, ComposerDiffusionTransformer)

        # Move text encoder to device
        self.text_encoder = self.text_encoder.cuda()
        # Generate images
        gen_images = []
        with get_precision_context(state.precision):
            for prompt in self.prompts:
                input_coords, input_mask = self._create_input_coords_and_mask(self.height, self.width)
                conditioning, conditioning_coords, conditioning_mask = self._encode_prompt(prompt)
                gen_patches = model.generate(input_coords=input_coords,
                                             input_mask=input_mask,
                                             conditioning=conditioning,
                                             conditioning_coords=conditioning_coords,
                                             conditioning_mask=conditioning_mask,
                                             guidance_scale=self.guidance_scale,
                                             num_timesteps=self.num_timesteps,
                                             progress_bar=False,
                                             seed=self.seed)
                gen_image = self._reconstruct_image_from_patches(gen_patches[0], input_coords[0])
                gen_images.append(gen_image)
        # Move text encoder back off device
        self.text_encoder = self.text_encoder.cpu()

        # Log images to wandb
        for prompt, image in zip(self.prompts, gen_images):
            logger.log_images(images=image, name=prompt, step=state.timestamp.batch.value, use_table=False)
