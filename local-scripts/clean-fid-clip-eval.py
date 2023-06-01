# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Eval script for using clean-fid."""

import argparse
import os

import wandb
from cleanfid import fid
from composer import Trainer
from composer.loggers import WandBLogger
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from diffusion.callbacks import LogDiffusionImages
from diffusion.datasets import build_streaming_cocoval_dataloader
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

args = parser.parse_args()

# Verify output dirs exist, if they don't, create them
if not os.path.exists(args.real_image_path):
    os.makedirs(args.real_image_path)
if not os.path.exists(args.gen_image_path):
    os.makedirs(args.gen_image_path)

# Create the eval dataloader
coco_val_dataloader = build_streaming_cocoval_dataloader(
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

# If a checkpoint is specified, evaluate it. Otherwise evaluate the pretrained SD2.0 model.
pretrained = args.load_path is None
if pretrained:
    name = args.name + '-pretrained'
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
trainer = Trainer(model=model, load_path=args.load_path, load_weights_only=True, eval_dataloader=coco_val_dataloader)

# Iterate over the coco dataloader
for batch_id, batch in tqdm(enumerate(coco_val_dataloader)):
    real_images = batch['image']
    captions = batch['captions']

    # Generate images from the captions
    generated_images = model.generate(tokenized_prompts=captions,
                                      height=args.size,
                                      width=args.size,
                                      guidance_scale=args.guidance_scale,
                                      seed=args.seed,
                                      progress_bar=False)
    # Save the real images
    for i, img in enumerate(real_images):
        img_id = args.batch_size * batch_id + i
        to_pil_image(img).save(f'{args.real_image_path}/{img_id}.png')
    # Save the generated images
    for i, img in enumerate(generated_images):
        img_id = args.batch_size * batch_id + i
        to_pil_image(img).save(f'{args.gen_image_path}/{img_id}.png')
    break

# Compute FID
score = fid.compute_fid(args.real_image_path, args.gen_image_path)
print(f'FID: {score}')

# Optionally log to wandb
if args.wandb:
    wandb.init(project=args.project, name=args.name)
    wandb.log({'metrics/FID': score})
