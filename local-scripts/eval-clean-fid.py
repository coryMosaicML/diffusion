# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Eval script for using clean-fid."""

import argparse
import json
import os
import sys

import clip
import torch
import wandb
from cleanfid import fid
from composer import Trainer
from composer.core import get_precision_context
from composer.loggers import WandBLogger
from composer.utils import dist
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from diffusion.callbacks import LogDiffusionImages
from diffusion.datasets import build_streaming_cocoval_dataloader, build_streaming_laion_dataloader
from diffusion.models import stable_diffusion_2

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def process_arguments(args):
    """Process command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', type=str, help='path to coco or laion streaming dataset(s)', nargs='+')
    parser.add_argument('--load_path', default=None, type=str, help='path to load model from')
    parser.add_argument('--guidance_scale',
                        default=[0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        type=float,
                        nargs='*',
                        help='guidance scale to evaluate at')
    parser.add_argument('--size', default=512, type=int, help='image size to evaluate at')
    parser.add_argument('--clip_model',
                        default='openai/clip-vit-base-patch16',
                        type=str,
                        help='CLIP model to use for CLIPScore')
    parser.add_argument('--no_crop', action='store_false', help='use resize instead of crop on COCO images.')
    parser.add_argument('--batch_size', default=16, type=int, help='eval batch size to use')
    parser.add_argument('--seed', default=17, type=int)
    parser.add_argument('-o', '--output_dir', default='/tmp/', type=str, help='output directory to save images to')
    parser.add_argument('--wandb', action='store_true', help='log to wandb')
    parser.add_argument('--project', default='diffusion-eval', type=str, help='wandb project to use')
    parser.add_argument('--name', default='fid-clip-evaluation', type=str, help='wandb name to use')
    parser.add_argument('--entity', default='mosaic-ml', type=str, help='wandb entity to use')
    parser.add_argument('--num_samples', default=30_000, type=int, help='number of samples to calculate FID on.')
    parser.add_argument('--precision', default='amp_fp16', type=str, choices=['fp32', 'amp_fp16', 'amp_bf16'])
    parser.add_argument(
        '--prompts',
        default=[
            'a couple waiting to cross the street underneath an umbrella.',
            'three men walking in the rain with umbrellas.',
            'a man is riding a red motor cycle, with baskets.',
            'a clock that has animal pictures instead of numbers.',
            'a brightly decorated bus sits on the road.',
            'a horse bucking with a rider on it, completely vertical, with another horse and onlookers.',
            'a white and blue bus is on a city street at night.',
            'a large clock tower on a building by a river',
            'beans and other food is sitting on a plate.',
            'a group of people that are standing up on a tennis court',
        ],
        type=str,
        nargs='*',
        help='Prompts used to generate images for visualization in wandb')
    args = parser.parse_args()
    return args


def make_dataloader(args):
    """Create the eval dataloader."""
    if 'coco' in args.remote[0]:
        eval_dataloader = build_streaming_cocoval_dataloader(
            remote=args.remote[0],  # COCO builder can only take one remote path
            local='/tmp/mds-cache/mds-coco-2014-val-fid-clip/',
            resize_size=args.size,
            use_crop=args.no_crop,
            batch_size=args.batch_size,
            prefetch_factor=2,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
    elif 'laion' in args.remote[0]:
        # Define a local directory for each remote path
        local = [f'/tmp/mds-cache/mds-laion/{i}' for i in range(len(args.remote))]
        eval_dataloader = build_streaming_laion_dataloader(
            remote=args.remote,
            local=local,
            resize_size=args.size,
            batch_size=args.batch_size,
            prefetch_factor=2,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            predownload=30_000,
            drop_last=False,
            shuffle=True,
        )
    return eval_dataloader


def load_checkpoint(args, eval_dataloader):
    """Load the model from a checkpoint."""
    pretrained = args.load_path is None

    model = stable_diffusion_2(
        model_name='stabilityai/stable-diffusion-2-base',
        val_metrics=[],
        val_guidance_scales=[],
        val_seed=args.seed,
        pretrained=pretrained,
        encode_latents_in_fp16=False,
        fsdp=False,
    )

    # Load model
    Trainer(model=model, load_path=args.load_path, load_weights_only=True, eval_dataloader=eval_dataloader)
    return model


def generate_images(args, model, eval_dataloader, guidance_scale, clip_metric):
    """Core image generation function."""
    # Verify output dirs exist, if they don't, create them
    real_image_path = os.path.join(args.output_dir, f'real_images_gs_{guidance_scale}')
    gen_image_path = os.path.join(args.output_dir, f'gen_images_gs_{guidance_scale}')
    if not os.path.exists(real_image_path) and dist.get_local_rank() == 0:
        os.makedirs(real_image_path)
    if not os.path.exists(gen_image_path) and dist.get_local_rank() == 0:
        os.makedirs(gen_image_path)

    # Reset the CLIP metric
    clip_metric.reset()

    # Storage for prompts
    prompts = {}
    # Iterate over the eval dataloader
    num_batches = len(eval_dataloader)
    for batch_id, batch in tqdm(enumerate(eval_dataloader)):
        # Break if enough samples were generated
        if batch_id * args.batch_size * dist.get_world_size() >= args.num_samples:
            break

        real_images = batch['image']
        captions = batch['captions']
        # Ensure a new seed for each batch, as randomness in model.generate is fixed.
        starting_seed = args.seed + num_batches * dist.get_local_rank()
        seed = starting_seed + batch_id
        # Generate images from the captions
        with get_precision_context(args.precision):
            generated_images = model.generate(tokenized_prompts=captions,
                                              height=args.size,
                                              width=args.size,
                                              guidance_scale=guidance_scale,
                                              seed=seed,
                                              progress_bar=False)
        # Get the prompts from the tokens
        text_captions = [model.tokenizer.decode(caption, skip_special_tokens=True) for caption in captions]
        clip_metric.update((generated_images * 255).to(torch.uint8), text_captions)
        # Save the real images
        for i, img in enumerate(real_images):
            to_pil_image(img).save(f'{real_image_path}/{batch_id}_{i}_rank_{dist.get_local_rank()}.png')
            prompts[f'{batch_id}_{i}_rank_{dist.get_local_rank()}'] = text_captions[i]
        # Save the generated images
        for i, img in enumerate(generated_images):
            to_pil_image(img).save(f'{gen_image_path}/{batch_id}_{i}_rank_{dist.get_local_rank()}.png')

    # Save the prompts as json
    json.dump(prompts, open(f'{real_image_path}/prompts_rank_{dist.get_local_rank()}.json', 'w'))


def compute_metrics(args, guidance_scale, clip_metric):
    """Compute metrics for the generated images."""
    # Path to find the generated images in
    real_image_path = os.path.join(args.output_dir, f'real_images_gs_{guidance_scale}')
    gen_image_path = os.path.join(args.output_dir, f'gen_images_gs_{guidance_scale}')

    metrics = {}
    # CLIP score
    clip_score = clip_metric.compute()
    metrics['CLIP-score'] = clip_score
    print(f'{guidance_scale} CLIP score: {clip_score}')

    # Need to tell clean-fid which device to use
    device = torch.device(dist.get_local_rank())
    # Standard FID
    fid_score = fid.compute_fid(real_image_path, gen_image_path, device=device, use_dataparallel=False, verbose=False)
    metrics['FID'] = fid_score
    print(f'{guidance_scale} FID: {fid_score}')
    # CLIP-FID from https://arxiv.org/abs/2203.06026
    clip_fid_score = fid.compute_fid(real_image_path,
                                     gen_image_path,
                                     mode='clean',
                                     model_name='clip_vit_b_32',
                                     device=device,
                                     use_dataparallel=False,
                                     verbose=False)
    metrics['CLIP-FID'] = clip_fid_score
    print(f'{guidance_scale} CLIP-FID: {clip_fid_score}')
    # KID
    kid_score = fid.compute_kid(real_image_path, gen_image_path, device=device, use_dataparallel=False, verbose=False)
    metrics['KID'] = kid_score
    print(f'{guidance_scale} KID: {kid_score}')
    return metrics


def generate_images_from_prompts(args, model, guidance_scale):
    """Generate images from prompts for visualization."""
    if args.prompts:
        with get_precision_context(args.precision):
            generated_images = model.generate(prompt=args.prompts,
                                              height=args.size,
                                              width=args.size,
                                              guidance_scale=guidance_scale,
                                              seed=args.seed)
    return generated_images


if __name__ == '__main__':
    args = process_arguments(sys.argv[1:])
    # Init wandb
    if args.wandb and dist.get_local_rank() == 0:
        wandb.init(name=args.name, project=args.project, entity=args.entity)
        wandb_logger = WandBLogger(name=args.name, project=args.project, entity=args.entity)
    # Create the eval dataloader
    eval_dataloader = make_dataloader(args)
    # Load the model
    model = load_checkpoint(args, eval_dataloader)
    # Create the clip metric
    device = dist.get_local_rank()
    clip_metric = CLIPScore(model_name_or_path=args.clip_model).to(device)
    # Predownload the CLIP model for computing clip-fid
    _, _ = clip.load('ViT-B/32', device=device)
    # Generate images and compute metrics for each guidance scale
    for guidance_scale in args.guidance_scale:
        dist.barrier()
        # Generate images and compute metrics
        generate_images(args=args,
                        model=model,
                        eval_dataloader=eval_dataloader,
                        guidance_scale=guidance_scale,
                        clip_metric=clip_metric)
        # Need to wait until all ranks have finished generating images before computing metrics
        dist.barrier()
        metrics = compute_metrics(args=args, guidance_scale=guidance_scale, clip_metric=clip_metric)
        # Generate images from prompts for visualization
        generated_images = generate_images_from_prompts(args=args, model=model, guidance_scale=guidance_scale)
        # Log metrics and images to wandb on rank 0
        if args.wandb and dist.get_local_rank() == 0:
            for metric, value in metrics.items():
                wandb.log({f'{guidance_scale}/{metric}': value})
            for prompt, image in zip(args.prompts, generated_images):
                wandb_logger.log_images(images=image, name=f'{prompt}_gs_{guidance_scale}')
