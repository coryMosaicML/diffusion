# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Constructors for diffusion models."""

from typing import List, Optional, Tuple

import torch
from composer.devices import DeviceGPU
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torchmetrics import MeanSquaredError
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig

from diffusion.models.autoencoder import AutoEncoderLoss, ComposerAutoEncoder, ComposerDiffusersAutoEncoder
from diffusion.models.pixel_diffusion import PixelDiffusion
from diffusion.models.stable_diffusion import StableDiffusion
from diffusion.schedulers.schedulers import ContinuousTimeScheduler

try:
    import xformers  # type: ignore
    del xformers
    is_xformers_installed = True
except:
    is_xformers_installed = False


def stable_diffusion_2(
    model_name: str = 'stabilityai/stable-diffusion-2-base',
    pretrained: bool = True,
    prediction_type: str = 'epsilon',
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    fsdp: bool = True,
):
    """Stable diffusion v2 training setup.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts.

    Args:
        model_name (str): Name of the model to load. Defaults to 'stabilityai/stable-diffusion-2-base'.
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError(), FrechetInceptionDistance(normalize=True)].
        val_guidance_scales (list, optional): List of scales to use for validation guidance. If None, defaults to
            [1.0, 3.0, 7.0].
        val_seed (int): Seed to use for generating evaluation images. Defaults to 1138.
        loss_bins (list, optional): List of tuples of (min, max) values to use for loss binning. If None, defaults to
            [(0, 1)].
        precomputed_latents (bool): Whether to use precomputed latents. Defaults to False.
        encode_latents_in_fp16 (bool): Whether to encode latents in fp16. Defaults to True.
        fsdp (bool): Whether to use FSDP. Defaults to True.
    """
    if train_metrics is None:
        train_metrics = [MeanSquaredError()]
    if val_metrics is None:
        val_metrics = [MeanSquaredError(), FrechetInceptionDistance(normalize=True)]
    if val_guidance_scales is None:
        val_guidance_scales = [1.0, 3.0, 7.0]
    if loss_bins is None:
        loss_bins = [(0, 1)]
    # Fix a bug where CLIPScore requires grad
    for metric in val_metrics:
        if isinstance(metric, CLIPScore):
            metric.requires_grad_(False)

    if pretrained:
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
    else:
        config = PretrainedConfig.get_config_dict(model_name, subfolder='unet')
        unet = UNet2DConditionModel(**config[0])

    if encode_latents_in_fp16:
        vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae', torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
    inference_noise_scheduler = DDIMScheduler(num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                                              beta_start=noise_scheduler.config.beta_start,
                                              beta_end=noise_scheduler.config.beta_end,
                                              beta_schedule=noise_scheduler.config.beta_schedule,
                                              trained_betas=noise_scheduler.config.trained_betas,
                                              clip_sample=noise_scheduler.config.clip_sample,
                                              set_alpha_to_one=noise_scheduler.config.set_alpha_to_one,
                                              prediction_type=prediction_type)

    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_guidance_scales=val_guidance_scales,
        val_seed=val_seed,
        loss_bins=loss_bins,
        precomputed_latents=precomputed_latents,
        encode_latents_in_fp16=encode_latents_in_fp16,
        fsdp=fsdp,
    )
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.unet.enable_xformers_memory_efficient_attention()
            model.vae.enable_xformers_memory_efficient_attention()
    return model


def stable_diffusion_xl(
    model_name: str = 'stabilityai/stable-diffusion-2-base',
    unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
    pretrained: bool = True,
    prediction_type: str = 'epsilon',
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    fsdp: bool = True,
):
    """Stable diffusion 2 training setup + SDXL UNet and VAE.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts. Currently uses UNet and VAE config from SDXL, but text encoder/tokenizer from SD2.

    Args:
        model_name (str): Name of the model to load. Determines the text encoder, tokenizer,
            and noise scheduler. Defaults to 'stabilityai/stable-diffusion-2-base'.
        unet_model_name (str): Name of the UNet model to load. Defaults to
            'stabilityai/stable-diffusion-xl-base-1.0'.
        vae_model_name (str): Name of the VAE model to load. Defaults to
            'madebyollin/sdxl-vae-fp16-fix' as the official VAE checkpoint (from
            'stabilityai/stable-diffusion-xl-base-1.0') is not compatible with fp16.
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError(), FrechetInceptionDistance(normalize=True)].
        val_guidance_scales (list, optional): List of scales to use for validation guidance. If None, defaults to
            [1.0, 3.0, 7.0].
        val_seed (int): Seed to use for generating evaluation images. Defaults to 1138.
        loss_bins (list, optional): List of tuples of (min, max) values to use for loss binning. If None, defaults to
            [(0, 1)].
        precomputed_latents (bool): Whether to use precomputed latents. Defaults to False.
        encode_latents_in_fp16 (bool): Whether to encode latents in fp16. Defaults to True.
        fsdp (bool): Whether to use FSDP. Defaults to True.
    """
    if train_metrics is None:
        train_metrics = [MeanSquaredError()]
    if val_metrics is None:
        val_metrics = [MeanSquaredError(), FrechetInceptionDistance(normalize=True)]
    if val_guidance_scales is None:
        val_guidance_scales = [1.0, 3.0, 7.0]
    if loss_bins is None:
        loss_bins = [(0, 1)]
    # Fix a bug where CLIPScore requires grad
    for metric in val_metrics:
        if isinstance(metric, CLIPScore):
            metric.requires_grad_(False)

    if pretrained:
        raise NotImplementedError('Full SDXL pipeline not implemented yet.')
    else:
        config = PretrainedConfig.get_config_dict(unet_model_name, subfolder='unet')
        # Currently not doing micro-conditioning, so set config appropriately
        config[0]['addition_embed_type'] = None
        config[0]['cross_attention_dim'] = 1024
        unet = UNet2DConditionModel(**config[0])

    # Prevent fsdp from wrapping up_blocks and down_blocks because the forward pass calls length on these
    unet.up_blocks._fsdp_wrap = False
    unet.down_blocks._fsdp_wrap = False
    for block in unet.up_blocks:
        block._fsdp_wrap = True
    for block in unet.down_blocks:
        block._fsdp_wrap = True

    if encode_latents_in_fp16:
        vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained(vae_model_name)
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
    inference_noise_scheduler = DDIMScheduler(num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                                              beta_start=noise_scheduler.config.beta_start,
                                              beta_end=noise_scheduler.config.beta_end,
                                              beta_schedule=noise_scheduler.config.beta_schedule,
                                              trained_betas=noise_scheduler.config.trained_betas,
                                              clip_sample=noise_scheduler.config.clip_sample,
                                              set_alpha_to_one=noise_scheduler.config.set_alpha_to_one,
                                              prediction_type=prediction_type)

    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_guidance_scales=val_guidance_scales,
        val_seed=val_seed,
        loss_bins=loss_bins,
        precomputed_latents=precomputed_latents,
        encode_latents_in_fp16=encode_latents_in_fp16,
        fsdp=fsdp,
    )
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.unet.enable_xformers_memory_efficient_attention()
            model.vae.enable_xformers_memory_efficient_attention()
    return model


def build_autoencoder(input_channels: int = 3,
                      output_channels: int = 3,
                      hidden_channels: int = 128,
                      latent_channels: int = 4,
                      double_latent_channels: bool = True,
                      channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
                      num_residual_blocks: int = 2,
                      use_conv_shortcut=False,
                      dropout_probability: float = 0.0,
                      resample_with_conv: bool = True,
                      zero_init_last: bool = False,
                      input_key: str = 'image',
                      learn_log_var: bool = True,
                      log_var_init: float = 0.0,
                      kl_divergence_weight: float = 1.0,
                      lpips_weight: float = 0.25,
                      discriminator_weight: float = 0.5,
                      discriminator_num_filters: int = 64,
                      discriminator_num_layers: int = 3):
    """Autoencoder training setup. By default, this config matches the network architecure used in SD2 and SDXL.

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        output_channels (int): Number of output channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        latent_channels (int): Number of latent channels. Default: `4`.
        double_latent_channels (bool): Whether to double the number of latent channels in the decoder. Default: `True`.
        channel_multipliers (tuple): Tuple of channel multipliers for each layer in the encoder and decoder. Default: `(1, 2, 4, 4)`.
        num_residual_blocks (int): Number of residual blocks in the encoder and decoder. Default: `2`.
        use_conv_shortcut (bool): Whether to use a convolutional shortcut in the residual blocks. Default: `False`.
        dropout_probability (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a convolutional resampling layer. Default: `True`.
        zero_init_last (bool): Whether to zero initialize the last layer in resblocks+discriminator. Default: `False`.
        input_key (str): Key to use for the input. Default: `image`.
        learn_log_var (bool): Whether to learn the output log variance in the VAE. Default: `True`.
        log_var_init (float): Initial value for the output log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPS loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
    """
    model = ComposerAutoEncoder(input_channels=input_channels,
                                output_channels=output_channels,
                                hidden_channels=hidden_channels,
                                latent_channels=latent_channels,
                                double_latent_channels=double_latent_channels,
                                channel_multipliers=channel_multipliers,
                                num_residual_blocks=num_residual_blocks,
                                use_conv_shortcut=use_conv_shortcut,
                                dropout_probability=dropout_probability,
                                resample_with_conv=resample_with_conv,
                                zero_init_last=zero_init_last,
                                input_key=input_key,
                                learn_log_var=learn_log_var,
                                log_var_init=log_var_init,
                                kl_divergence_weight=kl_divergence_weight,
                                lpips_weight=lpips_weight,
                                discriminator_weight=discriminator_weight,
                                discriminator_num_filters=discriminator_num_filters,
                                discriminator_num_layers=discriminator_num_layers)
    return model


def build_diffusers_autoencoder(model_name: str = 'stabilityai/stable-diffusion-2-base',
                                pretrained: bool = True,
                                vae_subfolder: bool = True,
                                output_channels: int = 3,
                                input_key: str = 'image',
                                learn_log_var: bool = True,
                                log_var_init: float = 0.0,
                                kl_divergence_weight: float = 1.0,
                                lpips_weight: float = 0.25,
                                discriminator_weight: float = 0.5,
                                discriminator_num_filters: int = 64,
                                discriminator_num_layers: int = 3,
                                zero_init_last: bool = False):
    """Diffusers autoencoder training setup.

    Args:
        model_name (str): Name of the Huggingface model. Default: `stabilityai/stable-diffusion-2-base`.
        pretrained (bool): Whether to use a pretrained model. Default: `True`.
        vae_subfolder: (bool): Whether to find the model config in a vae subfolder. Default: `True`.
        output_channels (int): Number of output channels. Default: `3`.
        input_key (str): Key for the input to the model. Default: `image`.
        learn_log_var (bool): Whether to learn the output log variance. Default: `True`.
        log_var_init (float): Initial value for the output log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPs loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the first layer of the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
    """
    # Get the model architecture and optionally the pretrained weights.
    if pretrained:
        if vae_subfolder:
            model = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
        else:
            model = AutoencoderKL.from_pretrained(model_name)
    else:
        if vae_subfolder:
            config = PretrainedConfig.get_config_dict(model_name, subfolder='vae')
        else:
            config = PretrainedConfig.get_config_dict(model_name)
        model = AutoencoderKL(**config[0])

    # Configure the loss function
    autoencoder_loss = AutoEncoderLoss(input_key=input_key,
                                       output_channels=output_channels,
                                       learn_log_var=learn_log_var,
                                       log_var_init=log_var_init,
                                       kl_divergence_weight=kl_divergence_weight,
                                       lpips_weight=lpips_weight,
                                       discriminator_weight=discriminator_weight,
                                       discriminator_num_filters=discriminator_num_filters,
                                       discriminator_num_layers=discriminator_num_layers)

    # Make the composer model
    composer_model = ComposerDiffusersAutoEncoder(model=model, loss_fn=autoencoder_loss, input_key=input_key)
    return composer_model


def discrete_pixel_diffusion(clip_model_name: str = 'openai/clip-vit-large-patch14', prediction_type='epsilon'):
    """Discrete pixel diffusion training setup.

    Args:
        clip_model_name (str, optional): Name of the clip model to load. Defaults to 'openai/clip-vit-large-patch14'.
        prediction_type (str, optional): Type of prediction to use. One of 'sample', 'epsilon', 'v_prediction'.
            Defaults to 'epsilon'.
    """
    # Create a pixel space unet
    unet = UNet2DConditionModel(in_channels=3,
                                out_channels=3,
                                attention_head_dim=[5, 10, 20, 20],
                                cross_attention_dim=768,
                                flip_sin_to_cos=True,
                                use_linear_projection=True)
    # Get the CLIP text encoder and tokenizer:
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    # Hard code the sheduler config
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule='scaled_linear',
                                    trained_betas=None,
                                    variance_type='fixed_small',
                                    clip_sample=False,
                                    prediction_type=prediction_type,
                                    thresholding=False,
                                    dynamic_thresholding_ratio=0.995,
                                    clip_sample_range=1.0,
                                    sample_max_value=1.0)
    inference_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                        beta_start=0.00085,
                                        beta_end=0.012,
                                        beta_schedule='scaled_linear',
                                        trained_betas=None,
                                        clip_sample=False,
                                        set_alpha_to_one=False,
                                        steps_offset=1,
                                        prediction_type=prediction_type,
                                        thresholding=False,
                                        dynamic_thresholding_ratio=0.995,
                                        clip_sample_range=1.0,
                                        sample_max_value=1.0)

    # Create the pixel space diffusion model
    model = PixelDiffusion(unet,
                           text_encoder,
                           tokenizer,
                           noise_scheduler,
                           inference_scheduler=inference_scheduler,
                           prediction_type=prediction_type,
                           train_metrics=[MeanSquaredError()],
                           val_metrics=[MeanSquaredError()])

    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.model.enable_xformers_memory_efficient_attention()
    return model


def continuous_pixel_diffusion(clip_model_name: str = 'openai/clip-vit-large-patch14',
                               prediction_type='epsilon',
                               use_ode=False,
                               train_t_max=1.570795,
                               inference_t_max=1.56):
    """Continuous pixel diffusion training setup.

    Uses the same clip and unet config as `discrete_pixel_diffusion`, but operates in continous time as in the VP
    process in https://arxiv.org/abs/2011.13456.

    Args:
        clip_model_name (str, optional): Name of the clip model to load. Defaults to 'openai/clip-vit-large-patch14'.
        prediction_type (str, optional): Type of prediction to use. One of 'sample', 'epsilon', 'v_prediction'.
            Defaults to 'epsilon'.
        use_ode (bool, optional): Whether to do generation using the probability flow ODE. If not used, uses the
            reverse diffusion process. Defaults to False.
        train_t_max (float, optional): Maximum timestep during training. Defaults to 1.570795 (pi/2).
        inference_t_max (float, optional): Maximum timestep during inference.
            Defaults to 1.56 (pi/2 - 0.01 for stability).
    """
    # Create a pixel space unet
    unet = UNet2DConditionModel(in_channels=3,
                                out_channels=3,
                                attention_head_dim=[5, 10, 20, 20],
                                cross_attention_dim=768,
                                flip_sin_to_cos=True,
                                use_linear_projection=True)
    # Get the CLIP text encoder and tokenizer:
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    # Need to use the continuous time schedulers for training and inference.
    noise_scheduler = ContinuousTimeScheduler(t_max=train_t_max, prediction_type=prediction_type)
    inference_scheduler = ContinuousTimeScheduler(t_max=inference_t_max,
                                                  prediction_type=prediction_type,
                                                  use_ode=use_ode)

    # Create the pixel space diffusion model
    model = PixelDiffusion(unet,
                           text_encoder,
                           tokenizer,
                           noise_scheduler,
                           inference_scheduler=inference_scheduler,
                           prediction_type=prediction_type,
                           continuous_time=True,
                           train_metrics=[MeanSquaredError()],
                           val_metrics=[MeanSquaredError()])

    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.model.enable_xformers_memory_efficient_attention()
    return model
