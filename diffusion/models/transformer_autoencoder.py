# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Transformer based autoencoder parts for training latent diffusion models."""

from typing import Dict

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import MeanMetric, MeanSquaredError, Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from diffusion.models.autoencoder import NlayerDiscriminator
from diffusion.models.layers import GradientScalingLayer
from diffusion.models.t2i_transformer import patchify, unpatchify
from diffusion.models.transformer import TransformerAutoEncoder


class TransformerAutoEncoderLoss(nn.Module):
    """Loss function for training a transformer based autoencoder. Includes discriminator.

    Args:
        input_key (str): Key for the input to the model. Default: `image`.
        ae_output_channels (int): Number of output channels. Default: `3`.
        learn_log_var (bool): Whether to learn the output log variance. Default: `True`.
        log_var_init (float): Initial value for the log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPs loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the first layer of the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
    """

    def __init__(self,
                 input_key: str = 'image',
                 ae_output_channels: int = 3,
                 lpips_weight: float = 0.25,
                 discriminator_weight: float = 0.5,
                 discriminator_num_filters: int = 64,
                 discriminator_num_layers: int = 3):
        super().__init__()
        self.input_key = input_key
        self.ae_output_channels = ae_output_channels
        self.lpips_weight = lpips_weight
        self.discriminator_weight = discriminator_weight
        self.discriminator_num_filters = discriminator_num_filters
        self.discriminator_num_layers = discriminator_num_layers
        self.learn_log_var = False

        # Set up LPIPs loss
        self.lpips = lpips.LPIPS(net='vgg').eval()
        # Only FSDP wrap models we are training
        self.lpips._fsdp_wrap = False
        # Ensure that lpips does not get trained
        for param in self.lpips.parameters():
            param.requires_grad_(False)
        for param in self.lpips.net.parameters():
            param.requires_grad_(False)

        # Set up the discriminator
        self.discriminator = NlayerDiscriminator(input_channels=self.ae_output_channels,
                                                 num_filters=self.discriminator_num_filters,
                                                 num_layers=self.discriminator_num_layers)
        self.scale_gradients = GradientScalingLayer()
        self.scale_gradients.register_full_backward_hook(self.scale_gradients.backward_hook)

    def set_discriminator_weight(self, weight: float):
        self.discriminator_weight = weight

    def calc_discriminator_adaptive_weight(self, ae_loss, fake_loss, last_layer):
        # Need to ensure the grad scale from the discriminator back to 1.0 to get the right norm
        self.scale_gradients.set_scale(1.0)
        # Get the grad norm from the non-adversarial loss
        ae_grads = torch.autograd.grad(ae_loss, last_layer, retain_graph=True)[0]
        # Get the grad norm for the discriminator loss
        disc_grads = torch.autograd.grad(fake_loss, last_layer, retain_graph=True)[0]
        # Calculate the updated discriminator weight based on the grad norms
        ae_grads_norm = torch.norm(ae_grads)
        disc_grads_norm = torch.norm(disc_grads)
        disc_weight = ae_grads_norm / (disc_grads_norm + 1e-4)
        disc_weight = torch.clamp(disc_weight, 0.0, 1e4).detach()
        disc_weight *= self.discriminator_weight
        # Set the discriminator weight. It should be negative to reverse gradients into the autoencoder.
        self.scale_gradients.set_scale(-disc_weight.item())
        return disc_weight, ae_grads_norm, disc_grads_norm

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                last_layer: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        # Basic L1 reconstruction loss
        ae_loss = F.l1_loss(outputs['x_recon'], batch[self.input_key])
        # Count the number of output elements to normalize the loss
        losses['ae_loss'] = ae_loss

        # LPIPs loss. Images for LPIPS must be in [-1, 1]
        recon_img = outputs['x_recon'].clamp(-1, 1)
        target_img = batch[self.input_key].clamp(-1, 1)
        lpips_loss = self.lpips(recon_img, target_img).mean()
        losses['lpips_loss'] = lpips_loss

        # Make the nll loss
        rec_loss = ae_loss + self.lpips_weight * lpips_loss
        losses['rec_loss'] = rec_loss

        # Discriminator loss
        real = self.discriminator(batch[self.input_key])
        fake = self.discriminator(self.scale_gradients(outputs['x_recon']))
        real_loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
        fake_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        losses['disc_real_loss'] = real_loss
        losses['disc_fake_loss'] = fake_loss
        losses['disc_loss'] = 0.5 * (real_loss + fake_loss)

        # Update the adaptive discriminator weight
        disc_weight, ae_grads_norm, disc_grads_norm = self.calc_discriminator_adaptive_weight(
            ae_loss, fake_loss, last_layer)
        losses['disc_weight'] = disc_weight
        losses['ae_grads_norm'] = ae_grads_norm
        losses['disc_grads_norm'] = disc_grads_norm

        # Combine the losses.
        total_loss = losses['rec_loss']
        total_loss += losses['disc_loss']
        losses['total'] = total_loss
        return losses


class ComposerTransformerAutoEncoder(ComposerModel):
    """Composer wrapper for the TransformerAutoEncoder.

    Args:
        model (TransformerAutoEncoder): AutoEncoder model to train.
        autoencoder_loss (TransformerAutoEncoderLoss): Auto encoder loss module.
        input_key (str): Key for the input to the model. Default: `image`.
        patch_size (int): Size of the patch to use for the transformer. This sets the downsampling factor for the
            autoencoder. Default: 8.
    """

    def __init__(self,
                 model: TransformerAutoEncoder,
                 autoencoder_loss: TransformerAutoEncoderLoss,
                 input_key: str = 'image',
                 patch_size: int = 8):
        super().__init__()
        self.model = model
        self.autoencoder_loss = autoencoder_loss
        self.input_key = input_key
        self.patch_size = patch_size

        # Set up train metrics
        train_metrics = [MeanSquaredError()]
        self.train_metrics = {metric.__class__.__name__: metric for metric in train_metrics}
        # Set up val metrics
        psnr_metric = PeakSignalNoiseRatio(data_range=2.0)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        val_metrics = [MeanSquaredError(), MeanMetric(), lpips_metric, psnr_metric, ssim_metric]
        self.val_metrics = {metric.__class__.__name__: metric for metric in val_metrics}

    def get_last_layer_weight(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.model.get_last_layer_weight()

    def forward(self, batch):
        image = batch[self.input_key]
        inputs, coords = patchify(image, self.patch_size)
        outputs = self.model(inputs, coords)
        x_recon, latents = outputs['x_recon'], outputs['latents']
        x_recon = [unpatchify(x, coords[i], patch_size=self.patch_size) for i, x in enumerate(x_recon)]
        x_recon = torch.stack(x_recon, dim=0)
        latents = [unpatchify(l, coords[i], patch_size=1) for i, l in enumerate(latents)]
        latents = torch.stack(latents, dim=0)
        return {'x_recon': x_recon, 'latents': latents}

    def loss(self, outputs, batch):
        last_layer = self.get_last_layer_weight()
        return self.autoencoder_loss(outputs, batch, last_layer)

    def eval_forward(self, batch, outputs=None):
        if outputs is not None:
            return outputs
        outputs = self.forward(batch)
        return outputs

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
        clamped_imgs = outputs['x_recon'].clamp(-1, 1)
        if isinstance(metric, MeanMetric):
            metric.update(torch.square(outputs['latents']))
        elif isinstance(metric, LearnedPerceptualImagePatchSimilarity):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, PeakSignalNoiseRatio):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, StructuralSimilarityIndexMeasure):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, MeanSquaredError):
            metric.update(outputs['x_recon'], batch[self.input_key])
        else:
            metric.update(outputs['x_recon'], batch[self.input_key])
