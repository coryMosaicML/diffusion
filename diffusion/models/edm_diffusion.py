# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion models."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError


class EDMDiffusion(ComposerModel):
    """Latent Diffusion ComposerModel.

    This is a Latent Diffusion model trained with the EDM framework.

    Args:
        unet (torch.nn.Module): HuggingFace conditional unet, must accept a
            (B, C, H, W) input, (B,) timestep array of noise timesteps,
            and (B, 77, 768) text conditioning vectors.
        vae (torch.nn.Module): HuggingFace or compatible vae.
            must support `.encode()` and `decode()` functions.
        text_encoder (torch.nn.Module): HuggingFace CLIP or LLM text enoder.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for
            text_encoder. For a `CLIPTextModel` this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        num_images_per_prompt (int): How many images to generate per prompt
            for evaluation. Default: `1`.
        latent_mean (Optional[tuple[float]]): The means of the latent space. If not specified, defaults to
            4 * (0.0,). Default: `None`.
        latent_std (Optional[tuple[float]]): The standard deviations of the latent space. If not specified,
            defaults to 4 * (1/0.13025,) for SDXL, or 4 * (1/0.18215,) for non-SDXL. Default: `None`.
        downsample_factor (int): The factor by which the image is downsampled by the autoencoder. Default `8`.
        train_metrics (list): List of torchmetrics to calculate during training.
            Default: `None`.
        val_metrics (list): List of torchmetrics to calculate during validation.
            Default: `None`.
        val_seed (int): Seed to use for generating eval images. Default: `1138`.
        image_key (str): The name of the image inputs in the dataloader batch.
            Default: `image_tensor`.
        caption_key (str): The name of the caption inputs in the dataloader batch.
            Default: `input_ids`.
        mask_pad_tokens (bool): whether to mask pad tokens in unet cross attention.
            Default: `False`.
    """

    def __init__(self,
                 unet,
                 vae,
                 text_encoder,
                 tokenizer,
                 latent_mean: Optional[Tuple[float]] = None,
                 latent_std: Optional[Tuple[float]] = None,
                 downsample_factor: int = 8,
                 train_metrics: Optional[List] = None,
                 val_metrics: Optional[List] = None,
                 val_seed: int = 1138,
                 image_key: str = 'image',
                 text_key: str = 'captions',
                 mask_pad_tokens: bool = False,
                 sigma_data: float = 1.0):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.downsample_factor = downsample_factor
        self.val_seed = val_seed
        self.image_key = image_key
        self.mask_pad_tokens = mask_pad_tokens
        # Prep latent stats
        if latent_mean is None:
            self.latent_mean = 4 * (0.0)
        if latent_std is None:
            self.latent_std = 4 * (1 / 0.13025,)
        self.latent_mean = torch.tensor(latent_mean).view(1, -1, 1, 1)
        self.latent_std = torch.tensor(latent_std).view(1, -1, 1, 1)

        self.train_metrics = train_metrics if train_metrics is not None else [MeanSquaredError()]
        self.val_metrics = val_metrics if val_metrics is not None else [MeanSquaredError()]
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.mask_pad_tokens = mask_pad_tokens

        # freeze text_encoder during diffusion training
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder = self.text_encoder.half()
        self.vae = self.vae.half()
        # only wrap models we are training
        self.text_encoder._fsdp_wrap = False
        self.vae._fsdp_wrap = False
        self.unet._fsdp_wrap = True
        # Optional rng generator
        self.rng_generator: Optional[torch.Generator] = None
        # EDM param
        self.sigma_data = sigma_data

    def _apply(self, fn):
        super(EDMDiffusion, self)._apply(fn)
        self.latent_mean = fn(self.latent_mean)
        self.latent_std = fn(self.latent_std)
        return self

    def set_rng_generator(self, rng_generator: torch.Generator):
        """Sets the rng generator for the model."""
        self.rng_generator = rng_generator

    def edm_forward(self, latents):
        """Perform forward diffusion according to the edm framework."""
        # First, sample noise levels according to a log-normal distribution
        log_sigma = 1.2 * (torch.randn(latents.shape[0], device=latents.device, generator=self.rng_generator) - 1)
        sigma = torch.exp(log_sigma).view(-1, 1, 1, 1)
        # Compute the EDM scaling factors
        c_in = (1 / torch.sqrt(self.sigma_data**2 + sigma**2))
        c_noise = log_sigma / 4
        c_skip = (self.sigma_data**2 / (self.sigma_data**2 + sigma**2))
        c_out = (sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2))
        # Then, add the noise to the latents according to EDM's forward process
        noise = torch.randn(*latents.shape, device=latents.device, generator=self.rng_generator)
        noised_latents = latents + sigma * noise
        # Compute the model inputs using the EDM c_in scaling
        model_inputs = c_in * noised_latents
        # Compute the targets using the EDM c_out scaling
        targets = 1 / c_out * ((1 - c_skip) * latents - c_skip * sigma * noise)
        return model_inputs, c_noise, targets

    def forward(self, batch):
        latents, text_embeds, text_pooled_embeds, attention_mask, encoder_attention_mask = None, None, None, None, None
        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask']  # mask for text encoders
            # text mask for U-Net
            if self.mask_pad_tokens:
                encoder_attention_mask = _create_unet_attention_mask(attention_mask)

        inputs, conditionings = batch[self.image_key], batch[self.text_key]
        with torch.cuda.amp.autocast(enabled=False):
            # Encode the images to the latent space.
            latents = self.vae.encode(inputs.half())['latent_dist'].sample().data
            # Encode tokenized prompt into embedded text and pooled text embeddings
            text_encoder_out = self.text_encoder(conditionings, attention_mask=attention_mask)
            text_embeds = text_encoder_out[0]
            if len(text_encoder_out) <= 1:
                raise RuntimeError('EDMDiffusion model requires text encoder output to include a pooled text embedding')
            text_pooled_embeds = text_encoder_out[1]
        # Scale the latents
        latents = (latents - self.latent_mean) / self.latent_std

        # Zero dropped captions if needed
        if 'drop_caption_mask' in batch.keys():
            text_embeds *= batch['drop_caption_mask'].view(-1, 1, 1)
            if text_pooled_embeds is not None:
                text_pooled_embeds *= batch['drop_caption_mask'].view(-1, 1)

        # Perform forward diffusion according to the EDM framework
        model_inputs, c_noise, targets = self.edm_forward(latents)

        added_cond_kwargs = {}
        # Prepare added time ids & embeddings
        add_time_ids = torch.cat(
            [batch['cond_original_size'], batch['cond_crops_coords_top_left'], batch['cond_target_size']], dim=1)
        added_cond_kwargs = {'text_embeds': text_pooled_embeds, 'time_ids': add_time_ids}

        # Forward through the model
        return self.unet(model_inputs,
                         c_noise,
                         text_embeds,
                         encoder_attention_mask=encoder_attention_mask,
                         added_cond_kwargs=added_cond_kwargs)['sample'], targets

    def loss(self, outputs, batch):
        """Loss between unet output and targets."""
        return F.mse_loss(outputs[0], outputs[1])

    def eval_forward(self, batch, outputs=None):
        """For stable diffusion, eval forward computes unet outputs as well as some samples."""
        # Skip this if outputs have already been computed, e.g. during training
        if outputs is not None:
            return outputs
        return self.forward(batch)

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics
        metrics_dict = {metric.__class__.__name__: metric for metric in metrics}
        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        metric.update(outputs[0], outputs[1])

    def make_sampling_timesteps(self, N: int):
        rho = 7
        s_max_rho = 80**(1 / rho)
        s_min_rho = 0.002**(1 / rho)
        timesteps = [(s_max_rho + i / (N - 1) * (s_min_rho - s_max_rho))**(rho) for i in range(N)]
        return timesteps

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[list] = None,
        negative_prompt: Optional[list] = None,
        tokenized_prompts: Optional[torch.LongTensor] = None,
        tokenized_negative_prompts: Optional[torch.LongTensor] = None,
        tokenized_prompts_pad_mask: Optional[torch.LongTensor] = None,
        tokenized_negative_prompts_pad_mask: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 3.0,
        rescaled_guidance: Optional[float] = None,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = True,
        zero_out_negative_prompt: bool = True,
        crop_params: Optional[torch.Tensor] = None,
        input_size_params: Optional[torch.Tensor] = None,
    ):
        """Generates image from noise.

        Performs the backward diffusion process, each inference step takes
        one forward pass through the unet.

        Args:
            prompt (str or List[str]): The prompt or prompts to guide the image generation.
            negative_prompt (str or List[str]): The prompt or prompts to guide the
                image generation away from. Ignored when not using guidance
                (i.e., ignored if guidance_scale is less than 1).
                Must be the same length as list of prompts. Default: `None`.
            tokenized_prompts (torch.LongTensor): Optionally pass pre-tokenized prompts instead
                of string prompts. If SDXL, this will be a tensor of size [B, 2, max_length],
                otherwise will be of shape [B, max_length]. Default: `None`.
            tokenized_negative_prompts (torch.LongTensor): Optionally pass pre-tokenized negative
                prompts instead of string prompts. Default: `None`.
            tokenized_prompts_pad_mask (torch.LongTensor): Optionally pass padding mask for
                pre-tokenized prompts. Default `None`.
            tokenized_negative_prompts_pad_mask (torch.LongTensor): Optionall pass padding mask for
                pre-tokenized negative prompts. Default `None`.
            prompt_embeds (torch.FloatTensor): Optionally pass pre-tokenized prompts instead
                of string prompts. If both prompt and prompt_embeds
                are passed, prompt_embeds will be used. Default: `None`.
            negative_prompt_embeds (torch.FloatTensor): Optionally pass pre-embedded negative
                prompts instead of string negative prompts. If both negative_prompt and
                negative_prompt_embeds are passed, prompt_embeds will be used.  Default: `None`.
            height (int, optional): The height in pixels of the generated image.
                Default: `self.unet.config.sample_size * 8)`.
            width (int, optional): The width in pixels of the generated image.
                Default: `self.unet.config.sample_size * 8)`.
            num_inference_steps (int): The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense
                of slower inference. Default: `50`.
            guidance_scale (float): Guidance scale as defined in
                Classifier-Free Diffusion Guidance. guidance_scale is defined as w of equation
                2. of Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1.
                Higher guidance scale encourages to generate images that are closely linked
                to the text prompt, usually at the expense of lower image quality.
                Default: `3.0`.
            rescaled_guidance (float, optional): Rescaled guidance scale. If not specified, rescaled guidance will
                not be used. Default: `None`.
            num_images_per_prompt (int): The number of images to generate per prompt.
                 Default: `1`.
            progress_bar (bool): Whether to use the tqdm progress bar during generation.
                Default: `True`.
            seed (int): Random seed to use for generation. Set a seed for reproducible generation.
                Default: `None`.
            zero_out_negative_prompt (bool): Whether or not to zero out negative prompt if it is
                an empty string. Default: `True`.
            crop_params (torch.FloatTensor of size [Bx2], optional): Crop parameters to use
                when generating images with SDXL. Default: `None`.
            input_size_params (torch.FloatTensor of size [Bx2], optional): Size parameters
                (representing original size of input image) to use when generating images with SDXL.
                Default: `None`.
        """
        _check_prompt_given(prompt, tokenized_prompts, prompt_embeds)
        _check_prompt_lenths(prompt, negative_prompt)
        _check_prompt_lenths(tokenized_prompts, tokenized_negative_prompts)
        _check_prompt_lenths(prompt_embeds, negative_prompt_embeds)

        # Create rng for the generation
        device = self.vae.device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)  # type: ignore

        height = height or self.unet.config.sample_size * self.downsample_factor
        width = width or self.unet.config.sample_size * self.downsample_factor
        assert height is not None  # for type checking
        assert width is not None  # for type checking

        do_classifier_free_guidance = guidance_scale > 1.0  # type: ignore

        text_embeddings, pooled_text_embeddings, pad_attn_mask = self._prepare_text_embeddings(
            prompt, tokenized_prompts, tokenized_prompts_pad_mask, prompt_embeds, num_images_per_prompt)
        batch_size = len(text_embeddings)  # len prompts * num_images_per_prompt
        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
        pooled_embeddings, encoder_attn_mask = pooled_text_embeddings, pad_attn_mask
        if do_classifier_free_guidance:
            if not negative_prompt and not tokenized_negative_prompts and not negative_prompt_embeds and zero_out_negative_prompt:
                # Negative prompt is empty and we want to zero it out
                unconditional_embeddings = torch.zeros_like(text_embeddings)
                pooled_unconditional_embeddings = torch.zeros_like(pooled_text_embeddings)
                uncond_pad_attn_mask = torch.zeros_like(pad_attn_mask) if pad_attn_mask is not None else None
            else:
                if not negative_prompt:
                    negative_prompt = [''] * (batch_size // num_images_per_prompt)  # type: ignore
                unconditional_embeddings, pooled_unconditional_embeddings, uncond_pad_attn_mask = self._prepare_text_embeddings(
                    negative_prompt, tokenized_negative_prompts, tokenized_negative_prompts_pad_mask,
                    negative_prompt_embeds, num_images_per_prompt)

            # concat uncond + prompt
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])
            pooled_embeddings = torch.cat([pooled_unconditional_embeddings, pooled_text_embeddings])  # type: ignore
            if pad_attn_mask is not None:
                encoder_attn_mask = torch.cat([uncond_pad_attn_mask, pad_attn_mask])  # type: ignore
        else:
            pooled_embeddings = pooled_text_embeddings

        # prepare for diffusion generation process
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // self.downsample_factor,
             width // self.downsample_factor),
            device=device,
            generator=rng_generator,
        )

        added_cond_kwargs = {}
        # Prepare added time ids & embeddings
        if crop_params is None:
            crop_params = torch.zeros((batch_size, 2), dtype=text_embeddings.dtype)
        if input_size_params is None:
            input_size_params = torch.tensor([width, height], dtype=text_embeddings.dtype).repeat(batch_size, 1)
        output_size_params = torch.tensor([width, height], dtype=text_embeddings.dtype).repeat(batch_size, 1)

        if do_classifier_free_guidance:
            crop_params = torch.cat([crop_params, crop_params])
            input_size_params = torch.cat([input_size_params, input_size_params])
            output_size_params = torch.cat([output_size_params, output_size_params])

        add_time_ids = torch.cat([input_size_params, crop_params, output_size_params], dim=1).to(device)
        added_cond_kwargs = {'text_embeds': pooled_embeddings, 'time_ids': add_time_ids}

        # backward diffusion process
        timesteps = self.make_sampling_timesteps(num_inference_steps)
        # For EDM, latents need to be scaled up by the noise level
        latents = latents * timesteps[0]
        for i, t in enumerate(timesteps):
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            log_sigma = torch.log(t * torch.ones(latent_model_input.shape[0], device=device))
            sigma = (t * torch.ones(latents.shape[0], device=device)).view(-1, 1, 1, 1)
            # Compute the EDM scaling factors
            c_noise = log_sigma / 4
            c_in = 1 / torch.sqrt(self.sigma_data**2 + sigma**2)
            if do_classifier_free_guidance:
                c_in = torch.cat([c_in] * 2)
            c_skip = (self.sigma_data**2 / (self.sigma_data**2 + sigma**2))
            c_out = (sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2))
            # Model prediction
            scaled_model_input = c_in * latent_model_input
            pred = self.unet(scaled_model_input,
                             c_noise,
                             encoder_hidden_states=text_embeddings,
                             encoder_attention_mask=encoder_attn_mask,
                             added_cond_kwargs=added_cond_kwargs).sample

            if do_classifier_free_guidance:
                # perform guidance. Note this is only techincally correct for prediction_type 'epsilon'
                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
                # Optionally rescale the classifer free guidance
                if rescaled_guidance is not None:
                    std_pos = torch.std(pred_text, dim=(1, 2, 3), keepdim=True)
                    std_cfg = torch.std(pred, dim=(1, 2, 3), keepdim=True)
                    pred_rescaled = pred * (std_pos / std_cfg)
                    pred = pred_rescaled * rescaled_guidance + pred * (1 - rescaled_guidance)
            # Compute the predicted sample from the network output
            D_pred = c_skip * latents + c_out * pred
            # compute the previous noisy sample x_t -> x_t-1.
            if i < len(timesteps) - 1:
                delta_t = timesteps[i] - timesteps[(i + 1)]
            else:
                delta_t = timesteps[i]
            latents = latents - (latents - D_pred) * (delta_t / t)
        # We now use the vae to decode the generated latents back into the image.
        # scale and decode the image latents with vae
        latents = latents * self.latent_std + self.latent_mean
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)

    def _prepare_text_embeddings(self, prompt, tokenized_prompts, tokenized_pad_mask, prompt_embeds,
                                 num_images_per_prompt):
        """Tokenizes and embeds prompts if needed, then duplicates embeddings to support multiple generations per prompt."""
        device = self.text_encoder.device
        pooled_text_embeddings = None
        if prompt_embeds is None:
            if tokenized_prompts is None:
                tokenized_out = self.tokenizer(prompt,
                                               padding='max_length',
                                               max_length=self.tokenizer.model_max_length,
                                               truncation=True,
                                               return_tensors='pt')
                tokenized_prompts = tokenized_out['input_ids']
                tokenized_pad_mask = tokenized_out['attention_mask']
            if tokenized_pad_mask is not None:
                tokenized_pad_mask = tokenized_pad_mask.to(device)
            text_encoder_out = self.text_encoder(tokenized_prompts.to(device), attention_mask=tokenized_pad_mask)
            prompt_embeds = text_encoder_out[0]
            if len(text_encoder_out) <= 1:
                raise RuntimeError('EDM model requires text encoder output to include a pooled text embedding')
            pooled_text_embeddings = text_encoder_out[1]

        # duplicate text embeddings for each generation per prompt
        prompt_embeds = _duplicate_tensor(prompt_embeds, num_images_per_prompt)

        if not self.mask_pad_tokens:
            tokenized_pad_mask = None

        if tokenized_pad_mask is not None:
            tokenized_pad_mask = _create_unet_attention_mask(tokenized_pad_mask)
            tokenized_pad_mask = _duplicate_tensor(tokenized_pad_mask, num_images_per_prompt)

        pooled_text_embeddings = _duplicate_tensor(pooled_text_embeddings, num_images_per_prompt)
        return prompt_embeds, pooled_text_embeddings, tokenized_pad_mask


def _check_prompt_lenths(prompt, negative_prompt):
    if prompt is None and negative_prompt is None:
        return
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    if negative_prompt:
        negative_prompt_bs = 1 if isinstance(negative_prompt, str) else len(negative_prompt)
        if negative_prompt_bs != batch_size:
            raise ValueError('len(prompts) and len(negative_prompts) must be the same. \
                    A negative prompt must be provided for each given prompt.')


def _check_prompt_given(prompt, tokenized_prompts, prompt_embeds):
    if prompt is None and tokenized_prompts is None and prompt_embeds is None:
        raise ValueError('Must provide one of `prompt`, `tokenized_prompts`, or `prompt_embeds`')


def _create_unet_attention_mask(attention_mask):
    """Takes the union of multiple attention masks if given more than one mask."""
    if len(attention_mask.shape) == 2:
        return attention_mask
    elif len(attention_mask.shape) == 3:
        encoder_attention_mask = attention_mask[:, 0]
        for i in range(1, attention_mask.shape[1]):
            encoder_attention_mask |= attention_mask[:, i]
        return encoder_attention_mask
    else:
        raise ValueError(f'attention_mask should have either 2 or 3 dimensions: {attention_mask.shape}')


def _duplicate_tensor(tensor, num_images_per_prompt):
    """Duplicate tensor for multiple generations from a single prompt."""
    batch_size, seq_len = tensor.shape[:2]
    tensor = tensor.repeat(1, num_images_per_prompt, *[
        1,
    ] * len(tensor.shape[2:]))
    return tensor.view(batch_size * num_images_per_prompt, seq_len, *[
        -1,
    ] * len(tensor.shape[2:]))
