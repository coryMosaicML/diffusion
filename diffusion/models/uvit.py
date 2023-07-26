from composer.models import ComposerModel
from typing import List, Optional

import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint

import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError, Metric
from tqdm.auto import tqdm


if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


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


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, use_checkpoint=False,
                 clip_dim=768, num_clip_token=77, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.context_embed = nn.Linear(clip_dim, embed_dim)

        self.extras = 1 + num_clip_token

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, context):
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        context_token = self.context_embed(context)
        x = torch.cat((time_token, context_token, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x


class ComposerUViT(ComposerModel):

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
        return self.model(noised_inputs, timesteps, conditioning), targets, timesteps

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
            (batch_size, self.model.in_chans, height, width),
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
            pred = self.model(model_input, t_tensor, text_embeddings)

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
