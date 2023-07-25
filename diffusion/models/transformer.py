# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError, Metric
from tqdm.auto import tqdm


def modulate(x, shift, scale):
    """Modulate the input with the shift and scale"""
    return x * (1.0 + scale) + shift


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SelfAttention(nn.Module):
    """Standard self attention layer that supports masking"""

    def __init__(self, num_features, num_heads):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        # Linear layer to get q, k, and v
        self.qkv = nn.Linear(self.num_features, 3 * self.num_features)
        # Linear layer to get the output
        self.output_layer = nn.Linear(self.num_features, self.num_features)
        # Initialize all biases to zero
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.output_layer.bias)
        # Init the standard deviation of the weights to 0.02
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)

    def forward(self, x, mask=None):
        # Get the shape of the input
        B, T, C = x.size()
        # Calculate the query, key, and values all in one go
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # After this, q, k, and v will have shape (B, T, C)
        # Reshape the query, key, and values for multi-head attention
        # Also want to swap the sequence length and the head dimension for later matmuls
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        # Calculate the qk product. Don't forget to scale!
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, num_heads, T, T)
        # Apply the mask. Elements to be ignored should have a value of -inf.
        if mask is not None:
            qk = qk.masked_fill(mask == 0, float('-inf'))
        # Apply the softmax.
        attention_weights = torch.softmax(qk, dim=-1)  # (B, num_heads, T, T)
        # Apply the attention weights to the values.
        attention_out = torch.matmul(attention_weights, v)  # (B, num_heads, T, C // num_heads)
        # Swap the sequence length and the head dimension back and get rid of num_heads.
        attention_out = attention_out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        # Final linear layer to get the output
        out = self.output_layer(attention_out)
        return out


class DiTBlock(nn.Module):
    """Transformer block that supports masking"""

    def __init__(self, num_features, num_heads, expansion_factor=4):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        # Layer norm before the self attention
        self.layer_norm_1 = nn.LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)
        self.attention = SelfAttention(self.num_features, self.num_heads)
        # Layer norm before the MLP
        self.layer_norm_2 = nn.LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)
        # MLP layers. The MLP expands and then contracts the features.
        self.linear_1 = nn.Linear(self.num_features, self.expansion_factor * self.num_features)
        self.nonlinearity = nn.GELU(approximate='tanh')
        self.linear_2 = nn.Linear(self.expansion_factor * self.num_features, self.num_features)
        # MLP for the modulations
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.num_features, 6 * self.num_features, bias=True))
        # Initialize all biases to zero
        nn.init.zeros_(self.linear_1.bias)
        nn.init.zeros_(self.linear_2.bias)
        # Initialize the linear layer weights to have a standard deviation of 0.02
        nn.init.normal_(self.linear_1.weight, std=0.02)
        nn.init.normal_(self.linear_2.weight, std=0.02)
        # Initialize the modulations to zero. This will ensure the block acts as identity at initialization
        nn.init.zeros_(self.adaLN_mlp[1].weight)
        nn.init.zeros_(self.adaLN_mlp[1].bias)

    def forward(self, x, c, mask=None):
        # Calculate the modulations. Each is shape (B, num_features).
        mods = self.adaLN_mlp(c).unsqueeze(1).chunk(6, dim=2)
        # Forward, with modulations
        y = modulate(self.layer_norm_1(x), mods[0], mods[1])
        y = mods[2] * self.attention(y, mask=mask)
        x = x + y
        y = modulate(self.layer_norm_2(x), mods[3], mods[4])
        y = self.linear_1(y)
        y = self.nonlinearity(y)
        y = mods[5] * self.linear_2(y)
        x = x + y
        return x


class DiffusionTransformer(nn.Module):
    """Transformer model for diffusion"""

    def __init__(self,
                 num_features,
                 num_heads,
                 num_layers,
                 patch_size=16,
                 image_size=256,
                 cond_features_in=1024,
                 cond_timesteps=77,
                 num_timesteps=1000,
                 input_channels=3,
                 expansion_factor=4):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.image_size = image_size
        self.cond_features_in = cond_features_in
        self.cond_timesteps = cond_timesteps
        self.num_timesteps = num_timesteps
        self.input_channels = input_channels
        self.expansion_factor = expansion_factor

        # Embedding layer for the timesteps
        #self.timestep_embedding = nn.Embedding(self.num_timesteps, self.num_features)
        # Patching layer for images
        self.patch_embedding = nn.Conv2d(3, self.num_features, self.patch_size, stride=self.patch_size)
        # Init the patch embedding so it will add with positional embedding and respect scaling
        nn.init.kaiming_uniform_(self.patch_embedding.weight, nonlinearity='linear')
        nn.init.zeros_(self.patch_embedding.bias)
        self.patch_embedding.weight.data /= math.sqrt(2.0)
        # Embedding for the conditioning
        self.conditioning_embedding = nn.Linear(cond_features_in, self.num_features)
        # Init the conditioning embedding so it will add with positional embedding and respect scaling
        nn.init.kaiming_uniform_(self.conditioning_embedding.weight, nonlinearity='linear')
        nn.init.zeros_(self.conditioning_embedding.bias)
        self.conditioning_embedding.weight.data /= math.sqrt(2.0)
        # Embedding for the sequence positions
        self.patches_per_side = self.image_size // self.patch_size
        self.image_block_size = (self.image_size // self.patch_size)**2
        block_size = self.image_block_size + self.cond_timesteps
        self.position_embedding = nn.Embedding(block_size, self.num_features)
        # Init the position embedding so it will add with patch embedding and conditioning embedding
        self.position_embedding.weight.data /= math.sqrt(2.0)
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(self.num_features, self.num_heads, expansion_factor=self.expansion_factor)
            for _ in range(self.num_layers)
        ])
        # Output layer
        self.final_norm = nn.LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(self.num_features, self.input_channels * (self.patch_size**2))
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.num_features, 2 * self.num_features))
        # Init the output layer to zero
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        # Init the modulations to zero. This will ensure the block acts as identity at initialization
        nn.init.zeros_(self.adaLN_mlp[1].weight)
        nn.init.zeros_(self.adaLN_mlp[1].bias)

    def forward(self, x, t, conditioning=None, mask=None):
        # Embed the timestep
        t = timestep_embedding(t, self.num_features)
        # Patchify the image
        x = self.patch_embedding(x)
        # Flatten the image
        x = x.flatten(2).transpose(1, 2)  # (B, I, C)
        # Embed the conditioning
        c = self.conditioning_embedding(conditioning)  # (B, T, C)
        # Concatenate the image and the conditioning
        x = torch.cat([x, c], dim=1)  # (B, T+I, C)
        # Add the position embedding
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, T+I)
        position_embeddings = self.position_embedding(positions)  # (1, T+I, C)
        x = x + position_embeddings  # (B, T+I, C)
        # Pass through the transformer blocks
        for block in self.transformer_blocks:
            x = block(x, t, mask=mask)
        # Throw away the conditioning tokens
        x = x[:, 0:self.image_block_size, :]
        # Pass through the output layers to get the right number of elements
        mods = self.adaLN_mlp(t).unsqueeze(1).chunk(2, dim=2)
        x = modulate(self.final_norm(x), mods[0], mods[1])
        x = self.final_linear(x)
        # Convert the output back into a 2D image
        x = x.reshape(shape=(x.shape[0], self.patches_per_side, self.patches_per_side, self.patch_size, self.patch_size,
                             self.input_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], self.input_channels, self.image_size, self.image_size))
        return x


class ComposerDiffusionTransformer(ComposerModel):

    def __init__(self,
                 model,
                 text_encoder,
                 tokenizer,
                 noise_scheduler,
                 inference_noise_scheduler,
                 prediction_type: str = 'epsilon',
                 loss_fn=F.mse_loss,
                 train_metrics: Optional[List] = None,
                 val_metrics: Optional[List] = None,
                 val_seed: int = 1138,
                 val_guidance_scales: Optional[List] = None,
                 loss_bins: Optional[List] = None,
                 image_key: str = 'image',
                 text_key: str = 'captions',
                 fsdp: bool = True):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type.lower()
        if self.prediction_type not in ['sample', 'epsilon', 'v_prediction']:
            raise ValueError(f'prediction type must be one of sample, epsilon, or v_prediction. Got {prediction_type}')
        self.val_seed = val_seed
        self.image_key = image_key
        self.fsdp = fsdp

        # setup metrics
        if train_metrics is None:
            self.train_metrics = [MeanSquaredError()]
        else:
            self.train_metrics = train_metrics
        if val_metrics is None:
            val_metrics = [MeanSquaredError()]
        if val_guidance_scales is None:
            val_guidance_scales = [0.0]
        if loss_bins is None:
            loss_bins = [(0, 1)]
        # Create new val metrics for each guidance weight and each loss bin
        self.val_guidance_scales = val_guidance_scales

        # bin metrics
        self.val_metrics = {}
        metrics_to_sweep = ['FrechetInceptionDistance', 'InceptionScore', 'CLIPScore']
        for metric in val_metrics:
            if metric.__class__.__name__ in metrics_to_sweep:
                for scale in val_guidance_scales:
                    new_metric = type(metric)(**vars(metric))
                    # WARNING: ugly hack...
                    new_metric.guidance_scale = scale
                    scale_str = str(scale).replace('.', 'p')
                    self.val_metrics[f'{metric.__class__.__name__}-scale-{scale_str}'] = new_metric
            elif isinstance(metric, MeanSquaredError):
                for bin in loss_bins:
                    new_metric = type(metric)(**vars(metric))
                    # WARNING: ugly hack...
                    new_metric.loss_bin = bin
                    self.val_metrics[f'{metric.__class__.__name__}-bin-{bin[0]}-to-{bin[1]}'.replace('.',
                                                                                                     'p')] = new_metric
            else:
                self.val_metrics[metric.__class__.__name__] = metric
        # Add a mse metric for the full loss
        self.val_metrics['MeanSquaredError'] = MeanSquaredError()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.inference_scheduler = inference_noise_scheduler
        self.text_key = text_key
        # freeze text_encoder during diffusion training
        self.text_encoder.requires_grad_(False)

        if fsdp:
            # only wrap models we are training
            self.text_encoder._fsdp_wrap = False
            self.model._fsdp_wrap = True

    def forward(self, batch):
        inputs, conditioning = batch[self.image_key], batch[self.text_key]
        conditioning = conditioning.view(-1, conditioning.shape[-1])
        conditioning = self.text_encoder(conditioning)[0]
        # Sample the diffusion timesteps
        timesteps = torch.randint(0, len(self.noise_scheduler), (inputs.shape[0],), device=inputs.device)
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn_like(inputs)
        noised_inputs = self.noise_scheduler.add_noise(inputs, noise, timesteps)
        # Generate the targets
        if self.prediction_type == 'epsilon':
            targets = noise
        elif self.prediction_type == 'sample':
            targets = inputs
        elif self.prediction_type == 'v_prediction':
            targets = self.noise_scheduler.get_velocity(inputs, noise, timesteps)
        else:
            raise ValueError(
                f'prediction type must be one of sample, epsilon, or v_prediction. Got {self.prediction_type}')
        # Forward through the model
        return self.model(noised_inputs, timesteps, conditioning=conditioning), targets, timesteps

    def loss(self, outputs, batch):
        """Loss between unet output and added noise, typically mse."""
        return self.loss_fn(outputs[0], outputs[1])

    def eval_forward(self, batch, outputs=None):
        """Computes model outputs as well as some samples."""
        # Skip this if outputs have already been computed, e.g. during training
        if outputs is not None:
            return outputs
        # Get unet outputs
        model_out, targets, timesteps = self.forward(batch)
        # Sample images from the prompts in the batch
        prompts = batch[self.text_key]
        height, width = batch[self.image_key].shape[-2], batch[self.image_key].shape[-1]
        generated_images = {}
        for guidance_scale in self.val_guidance_scales:
            gen_images = self.generate(tokenized_prompts=prompts,
                                       height=height,
                                       width=width,
                                       guidance_scale=guidance_scale,
                                       seed=self.val_seed,
                                       progress_bar=False)
            generated_images[guidance_scale] = gen_images
        return model_out, targets, timesteps, generated_images

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        elif isinstance(metrics, list):
            metrics_dict = {metrics.__class__.__name__: metric for metric in metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        # If A MSE metric is associated with a loss bin, update the metric for the bin
        # Othewise, update the metric for the full loss
        if isinstance(metric, MeanSquaredError) and hasattr(metric, 'loss_bin'):
            # Get the loss bin from the metric
            loss_bin = metric.loss_bin
            # Get the loss for timesteps in the bin
            T_max = self.noise_scheduler.num_train_timesteps
            # Get the indices corresponding to timesteps in the bin
            bin_indices = torch.where(
                (outputs[2] >= loss_bin[0] * T_max) & (outputs[2] < loss_bin[1] * T_max))  # type: ignore
            # Update the metric for items in the bin
            metric.update(outputs[0][bin_indices], outputs[1][bin_indices])
        elif isinstance(metric, MeanSquaredError):
            metric.update(outputs[0], outputs[1])
        # FID metrics should be updated with the generated images at the desired guidance scale
        elif metric.__class__.__name__ == 'FrechetInceptionDistance':
            metric.update(batch[self.image_key], real=True)
            metric.update(outputs[3][metric.guidance_scale], real=False)
        # IS metrics should be updated with the generated images at the desired guidance scale
        elif metric.__class__.__name__ == 'InceptionScore':
            metric.update(outputs[3][metric.guidance_scale])
        # CLIP metrics should be updated with the generated images at the desired guidance scale
        elif metric.__class__.__name__ == 'CLIPScore':
            # Convert the captions to a list of strings
            captions = [self.tokenizer.decode(caption, skip_special_tokens=True) for caption in batch[self.text_key]]
            generated_images = (outputs[3][metric.guidance_scale] * 255).to(torch.uint8)
            metric.update(generated_images, captions)
        else:
            metric.update(outputs[0], outputs[1])

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[list] = None,
        negative_prompt: Optional[list] = None,
        tokenized_prompts: Optional[torch.LongTensor] = None,
        tokenized_negative_prompts: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 3.0,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = True,
    ):
        """Generates image from noise."""
        _check_prompt_given(prompt, tokenized_prompts, prompt_embeds)
        _check_prompt_lenths(prompt, negative_prompt)
        _check_prompt_lenths(tokenized_prompts, tokenized_negative_prompts)
        _check_prompt_lenths(prompt_embeds, negative_prompt_embeds)

        # Create rng for the generation
        device = next(self.model.parameters()).device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)  # type: ignore

        height = height or self.model.image_size
        width = width or self.model.image_size
        assert height is not None  # for type checking
        assert width is not None  # for type checking

        do_classifier_free_guidance = guidance_scale > 1.0  # type: ignore

        text_embeddings = self._prepare_text_embeddings(prompt, tokenized_prompts, prompt_embeds, num_images_per_prompt)
        batch_size = len(text_embeddings)  # len prompts * num_images_per_prompt
        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ([''] * (batch_size // num_images_per_prompt))  # type: ignore
            unconditional_embeddings = self._prepare_text_embeddings(negative_prompt, tokenized_negative_prompts,
                                                                     negative_prompt_embeds, num_images_per_prompt)
            # concat uncond + prompt
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])

        # prepare for diffusion generation process
        inputs = torch.randn(
            (batch_size, self.model.input_channels, height, width),
            device=device,
            generator=rng_generator,
        )

        self.inference_scheduler.set_timesteps(num_inference_steps)
        # scale the initial noise by the standard deviation required by the scheduler
        inputs = inputs * self.inference_scheduler.init_noise_sigma

        # backward diffusion process
        for t in tqdm(self.inference_scheduler.timesteps, disable=not progress_bar):
            if do_classifier_free_guidance:
                model_input = torch.cat([inputs] * 2)
            else:
                model_input = inputs

            model_input = self.inference_scheduler.scale_model_input(model_input, t)
            # Model prediction
            t_tensor = torch.ones(model_input.shape[0], dtype=torch.int64, device=device) * t
            pred = self.model(model_input, t_tensor, conditioning=text_embeddings)

            if do_classifier_free_guidance:
                # perform guidance
                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            inputs = self.inference_scheduler.step(pred, t, inputs, generator=rng_generator).prev_sample

        # We now decode back into an image
        image = (inputs / 2 + 0.5).clamp(0, 1)
        return image.detach()  # (batch*num_images_per_prompt, channel, h, w)

    def _prepare_text_embeddings(self, prompt, tokenized_prompts, prompt_embeds, num_images_per_prompt):
        """Tokenizes and embeds prompts if needed, then duplicates embeddings to support multiple generations per prompt."""
        device = self.text_encoder.device
        if prompt_embeds is None:
            if tokenized_prompts is None:
                tokenized_prompts = self.tokenizer(prompt,
                                                   padding='max_length',
                                                   max_length=self.tokenizer.model_max_length,
                                                   truncation=True,
                                                   return_tensors='pt').input_ids
            text_embeddings = self.text_encoder(tokenized_prompts.to(device))[0]  # type: ignore
        else:
            text_embeddings = prompt_embeds

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)  # type: ignore
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        return text_embeddings


def _check_prompt_lenths(prompt, negative_prompt):
    if prompt is None and negative_prompt is None:
        return
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    if negative_prompt:
        negative_prompt_bs = 1 if isinstance(negative_prompt, str) else len(negative_prompt)
        if negative_prompt_bs != batch_size:
            raise ValueError(f'len(prompts) and len(negative_prompts) must be the same. \
                    A negative prompt must be provided for each given prompt.')


def _check_prompt_given(prompt, tokenized_prompts, prompt_embeds):
    if prompt is None and tokenized_prompts is None and prompt_embeds is None:
        raise ValueError('Must provide one of `prompt`, `tokenized_prompts`, or `prompt_embeds`')
