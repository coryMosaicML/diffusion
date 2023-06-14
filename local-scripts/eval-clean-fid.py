# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Eval script for using clean-fid."""

import argparse
import os

import torch
import wandb
from cleanfid import fid
from composer import Trainer
from composer.core import get_precision_context
from composer.loggers import WandBLogger
from composer.utils import dist
from torchmetrics.multimodal import CLIPScore
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from diffusion.callbacks import LogDiffusionImages
from diffusion.datasets import build_streaming_cocoval_dataloader, build_streaming_laion_dataloader
from diffusion.models import stable_diffusion_2

parser = argparse.ArgumentParser()
parser.add_argument('--remote', type=str, help='path to coco or laion streaming dataset(s)', nargs='+')
parser.add_argument('--load_path', default=None, type=str, help='path to load model from')
parser.add_argument('--guidance_scale', default=1.0, type=float, help='guidance scale to evaluate at')
parser.add_argument('--size', default=512, type=int, help='image size to evaluate at')
parser.add_argument('--no_crop', action='store_false', help='use resize instead of crop on COCO images.')
parser.add_argument('--batch_size', default=16, type=int, help='eval batch size to use')
parser.add_argument('--seed', default=17, type=int)
parser.add_argument('-r', '--real_image_path', type=str, default='/tmp/real-images', help='path to store real images')
parser.add_argument('-g',
                    '--gen_image_path',
                    type=str,
                    default='/tmp/gen-images',
                    help='path to store generated images')

parser.add_argument('--wandb', action='store_true', help='log to wandb')
parser.add_argument('--project', default='diffusion-eval', type=str, help='wandb project to use')
parser.add_argument('--name', default='fid-clip-evaluation', type=str, help='wandb name to use')
parser.add_argument('--entity', default='mosaic-ml', type=str, help='wandb entity to use')
parser.add_argument('--num_samples', default=30_000, type=int, help='number of samples to calculate FID on.')
parser.add_argument('--precision', default='amp_fp16', type=str, choices=['fp32', 'amp_fp16', 'amp_bf16'])
parser.add_argument('--prompts',
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

# Verify output dirs exist, if they don't, create them
if not os.path.exists(args.real_image_path) and dist.get_local_rank() == 0:
    os.makedirs(args.real_image_path)
if not os.path.exists(args.gen_image_path) and dist.get_local_rank() == 0:
    os.makedirs(args.gen_image_path)

# Create the eval dataloader
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

# If a checkpoint is specified, evaluate it. Otherwise evaluate the pretrained SD2.0 model.
pretrained = args.load_path is None
if pretrained:
    name = args.name + '-pretrained'
else:
    name = args.name + '-checkpoint'
name += f'-{args.guidance_scale}-{args.seed}'
print(f'Evaluating {name}')
# Init wandb
if args.wandb and dist.get_local_rank() == 0:
    wandb.init(name=name, project=args.project, entity=args.entity)
    wandb_logger = WandBLogger(name=name, project=args.project, entity=args.entity)

model = stable_diffusion_2(
    model_name='stabilityai/stable-diffusion-2-base',
    val_metrics=[],
    val_guidance_scales=[],
    val_seed=args.seed,
    pretrained=pretrained,
    encode_latents_in_fp16=False,
    fsdp=False,
)

clip_score = CLIPScore()

# Load model
Trainer(model=model, load_path=args.load_path, load_weights_only=True, eval_dataloader=eval_dataloader)

# Iterate over the eval dataloader
num_batches = len(eval_dataloader)
for batch_id, batch in tqdm(enumerate(eval_dataloader)):
    # Break if enough samples were generated
    if batch_id * args.batch_size * dist.get_world_size() >= args.num_samples:
        break

    real_images = batch['image']
    captions = batch['captions']
    # Ensure a new seed for each batch
    starting_seed = args.seed + num_batches * dist.get_local_rank()
    seed = starting_seed + batch_id
    # Generate images from the captions
    with get_precision_context(args.precision):
        generated_images = model.generate(tokenized_prompts=captions,
                                          height=args.size,
                                          width=args.size,
                                          guidance_scale=args.guidance_scale,
                                          seed=seed,
                                          progress_bar=False)
    # Save the real images
    for i, img in enumerate(real_images):
        to_pil_image(img).save(f'{args.real_image_path}/{batch_id}_{i}_rank_{dist.get_local_rank()}.png')
    # Save the generated images
    for i, img in enumerate(generated_images):
        to_pil_image(img).save(f'{args.gen_image_path}/{batch_id}_{i}_rank_{dist.get_local_rank()}.png')

    # Calculate CLIPScore
    captions = [model.tokenizer.decode(caption, skip_special_tokens=True) for caption in captions]
    scaled_gen_imgs = (generated_images * 255).to(torch.uint8)
    clip_score.update(scaled_gen_imgs, captions)

# Need to wait until all processes have finished generating images
dist.barrier()

# Compute metrics and log images
if dist.get_local_rank() == 0:
    # Standard FID
    fid_score = fid.compute_fid(args.real_image_path, args.gen_image_path)
    print(f'{name} FID: {fid_score}')
    # CLIP-FID from https://arxiv.org/abs/2203.06026
    clip_fid_score = fid.compute_fid(args.real_image_path,
                                     args.gen_image_path,
                                     mode='clean',
                                     model_name='clip_vit_b_32')
    print(f'{name} CLIP-FID: {clip_fid_score}')
    # KID
    kid_score = fid.compute_kid(args.real_image_path, args.gen_image_path)
    print(f'{name} KID: {kid_score}')
    # CLIP Score
    clip_score_val = clip_score.compute()
    print(f'{name} CLIP Score: {clip_score_val}')
    # Optionally log to wandb
    if args.wandb:
        wandb.log({'metrics/FID': fid_score})
        wandb.log({'metrics/CLIP-FID': clip_fid_score})
        wandb.log({'metrics/KID': kid_score})
        wandb.log({'metrics/CLIP-Score': clip_score_val})

        # Generate images based on args.prompts
        with get_precision_context(args.precision):
            generated_images = model.generate(prompt=args.prompts,
                                              height=args.size,
                                              width=args.size,
                                              guidance_scale=args.guidance_scale,
                                              seed=seed)
            for prompt, image in zip(args.prompts, generated_images):
                wandb_logger.log_images(images=image, name=prompt)

dist.barrier()
