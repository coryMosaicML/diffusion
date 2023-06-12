# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Eval script for using clean-fid."""

import argparse
import os

import wandb
from cleanfid import fid
from composer import Trainer
from composer.loggers import WandBLogger
from composer.utils import dist
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from diffusion.callbacks import LogDiffusionImages
from diffusion.datasets import build_streaming_cocoval_dataloader, build_streaming_laion_dataloader
from diffusion.models import stable_diffusion_2

parser = argparse.ArgumentParser()
parser.add_argument('--remote', type=str, help='path to coco streaming dataset')
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
args = parser.parse_args()

# Verify output dirs exist, if they don't, create them
if not os.path.exists(args.real_image_path):
    os.makedirs(args.real_image_path)
if not os.path.exists(args.gen_image_path):
    os.makedirs(args.gen_image_path)

# Create the eval dataloader
if 'coco' in args.remote:
    eval_dataloader = build_streaming_cocoval_dataloader(
        remote=args.remote,
        local='/tmp/mds-cache/mds-coco-2014-val-fid-clip/',
        resize_size=args.size,
        use_crop=args.no_crop,
        batch_size=args.batch_size,
        prefetch_factor=2,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
elif 'laion' in args.remote:
    eval_dataloader = build_streaming_laion_dataloader(
        remote=args.remote,
        local='/tmp/mds-cache/mds-laion/',
        batch_size=args.batch_size,
        resize_size=args.size,
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
trainer = Trainer(model=model, load_path=args.load_path, load_weights_only=True, eval_dataloader=eval_dataloader)

# Iterate over the coco dataloader
num_batches = len(eval_dataloader)
for batch_id, batch in tqdm(enumerate(eval_dataloader)):
    if batch_id >= args.num_samples:
        break
    real_images = batch['image']
    captions = batch['captions']
    # Ensure a new seed for each batch
    starting_seed = args.seed + num_batches * dist.get_local_rank()
    seed = starting_seed + batch_id
    # Generate images from the captions
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

# Need to wait until all processes have finished generating images
dist.barrier()

# Compute metrics
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
    # Optionally log to wandb
    if args.wandb:
        wandb.log({'metrics/FID': fid_score})
        wandb.log({'metrics/CLIP-FID': clip_fid_score})
        wandb.log({'metrics/KID': kid_score})
