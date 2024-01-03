# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Filtering script for pulling high quality photos from the MDS dataset."""

import argparse
import os

import clip
import imagehash
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX  # type: ignore
from llava.conversation import SeparatorStyle, conv_templates  # type: ignore
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # type: ignore
from llava.model.builder import load_pretrained_model  # type: ignore
from llava.utils import disable_torch_init  # type: ignore
from PIL import Image
from streaming import Stream
from streaming.base import MDSWriter
from torchvision import transforms

from diffusion.datasets.image_caption import StreamingImageCaptionDataset

parser = argparse.ArgumentParser()
parser.add_argument('--remotes', nargs='+', help='List of remotes to use for the dataset.')
parser.add_argument('--locals', nargs='+', help='List of local directories to use for the dataset.')
parser.add_argument('--output', help='Output path for the filtered dataset.')
parser.add_argument('--image_key', type=str, default='jpg', help='Dataset image key.')
parser.add_argument('--caption_key', type=str, default='jpg', help='Dataset caption key.')
parser.add_argument('--aesthetics_threshold', type=float, default=6.5, help='Aesthetics threshold for filtering.')
parser.add_argument('--generate_caption', action='store_true', help='Whether to generate a caption for the image.')
parser.add_argument('--max_size', type=int, default=1024, help='Maximum image side length.')
parser.add_argument('--start', type=int, default=0, help='Index of sample to start with.')
parser.add_argument('--end', type=int, default=10000, help='Index of sample to end with.')
args = parser.parse_args()


# load Aesthetics model
class MLP(torch.nn.Module):
    """From https://github.com/christophschuhmann/improved-aesthetic-predictor."""

    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


class DatasetFilter():
    """Filter a dataset based on a set of model predictions."""

    def __init__(self,
                 remotes,
                 locals,
                 output,
                 image_key: str = 'jpg',
                 caption_key: str = 'caption',
                 aesthetics_threshold: float = 5.0,
                 generate_caption: bool = True,
                 max_size: int = 1024,
                 start: int = 0,
                 end: int = 10000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.remotes = remotes
        self.locals = locals
        self.output = output
        self.image_key = image_key
        self.caption_key = caption_key
        self.aesthetics_threshold = aesthetics_threshold
        self.generate_caption = generate_caption
        self.max_size = max_size
        self.start = start
        self.end = end
        # Make the dataset
        self.dataset = self._make_dataset()
        # Load the aesthetics v2 model
        self.preprocess, self.clip_model, self.aesthetics_model = self._load_aesthetics_v2()
        # Load the llava model
        self.llava_tokenizer, self.llava_model, self.llava_image_processor, self.llava_context_len = self._load_llava()
        self.conv_mode = 'llava_v1'
        self.to_tensor = transforms.ToTensor()
        # Storage for hashes
        self.hashes = {}
        # Fields for the MDSWriter
        if self.generate_caption:
            self.fields = {'image': 'pil', 'captions': 'json', 'llava_caption': 'json', 'aesthetics_score': 'float32'}
        else:
            self.fields = {'image': 'pil', 'captions': 'json', 'aesthetics_score': 'float32'}

    def _make_dataset(self):
        streams = []
        for r, l in zip(self.remotes, self.locals):
            streams.append(Stream(remote=r, local=l))

        transform = transforms.Compose([])
        dataset = StreamingImageCaptionDataset(
            streams=streams,
            tokenizer_name_or_path='stabilityai/stable-diffusion-2-base',
            caption_selection='first',
            image_key=self.image_key,
            caption_key=self.caption_key,
            crop=None,
            download_timeout=120,
            transform=transform,
            num_canonical_nodes=1,
        )
        return dataset

    def _resize_image(self, image):
        """Resize an image such that the largest dimension is no bigger than size."""
        # Shortcut to avoid resizing
        if max(image.size) <= self.max_size:
            return image

        if image.size[0] > image.size[1]:
            # If width is bigger than height, resize to width = size
            height_size = int(self.max_size * image.size[1] / image.size[0])
            resized_image = image.resize((self.max_size, height_size), Image.ANTIALIAS)
        elif image.size[0] < image.size[1]:
            # If height is bigger than width, resize to height = size
            width_size = int(self.max_size * image.size[0] / image.size[1])
            resized_image = image.resize((width_size, self.max_size), Image.ANTIALIAS)
        else:
            # If height and width are equal, resize to size x size
            resized_image = image.resize((self.max_size, self.max_size), Image.ANTIALIAS)
        return resized_image

    def _load_aesthetics_v2(self):
        """Loads the aesthetics v2 model."""
        aesthetics_model = MLP(768)
        s = torch.load('sac+logos+ava1-l14-linearMSE.pth')
        aesthetics_model.load_state_dict(s)
        aesthetics_model.to(self.device)
        aesthetics_model.eval()
        clip_model, preprocess = clip.load('ViT-L/14', device=self.device)
        return preprocess, clip_model, aesthetics_model

    def _normalized(self, a, axis=-1, order=2):
        """Utility function to normalize the rows of a numpy array."""
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def get_aesthetics_score(self, image: Image.Image):
        """Get the aesthetics score for an image."""
        resized_image = self._resize_image(image)
        resized_image = self.preprocess(resized_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(resized_image)
        im_emb_arr = self._normalized(image_features.cpu().detach().numpy())
        prediction = self.aesthetics_model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.float32))
        return prediction.item()

    def _load_llava(self):
        """Loads the llava model."""
        # Download the llava model if it isn't there already.
        snapshot_download(
            repo_id='liuhaotian/llava-v1.5-13b',
            local_dir='/tmp/llava',
            local_dir_use_symlinks=False,
        )
        disable_torch_init()
        model_path = os.path.expanduser('/tmp/llava')
        model_name = get_model_name_from_path(model_path)
        return load_pretrained_model(model_path, None, model_name)

    def _llava_add_image_tokens(self, prompt: str) -> str:
        if self.llava_model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        return prompt

    def _llava_tokenize(self, prompt: str) -> torch.Tensor:
        input_ids = tokenizer_image_token(prompt, self.llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids.unsqueeze(0).to(self.device)

    def _get_llava_output(self, image: Image.Image, prompt: str) -> str:
        """Get the output from llava."""
        # Format the prompt
        prompt = self._llava_add_image_tokens(prompt)
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self._llava_tokenize(prompt)
        # Make the stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # Prep the image
        img = self.to_tensor(image).unsqueeze(0).to(self.device)  # In range (0, 1)
        image_tensor = self.llava_image_processor.preprocess(img, do_rescale=False, return_tensors='pt')['pixel_values']
        # Forward through the model
        with torch.inference_mode():
            output_ids = self.llava_model.generate(input_ids,
                                                   images=image_tensor.half().to(self.device),
                                                   do_sample=True,
                                                   temperature=0.2,
                                                   top_p=None,
                                                   num_beams=1,
                                                   max_new_tokens=1024,
                                                   use_cache=True)
        # Postprocess outputs
        input_token_len = input_ids.shape[1]
        output_text = self.llava_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        output_text = output_text.strip()
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)]
        output_text = output_text.strip()
        return output_text

    def _llava_classify_photo(self, image):
        llava_image = self._resize_image(image)
        llava_output = self._get_llava_output(llava_image, 'Is this image a photo? Respond with yes or no.')
        return llava_output.lower() == 'yes'

    def _llava_classify_watermark(self, image):
        llava_image = self._resize_image(image)
        llava_output = self._get_llava_output(llava_image, 'Does this image have a watermark? Respond with yes or no.')
        return llava_output.lower() == 'yes'

    def _llava_caption(self, image, alt_text=None):
        # The pixart-alpha query plus extra alt text info and a request to not use complete sentences.
        #query = "Describe this image and its style in a very detailed manner."
        query = 'Write a detailed caption.'
        if alt_text is not None:
            #query += f" Include specific information from the following as needed: {alt_text}"
            query += f' Include nouns from the following if relevant: {alt_text}'
        query += ' Do not use complete sentences.'

        llava_image = self._resize_image(image)
        llava_output = self._get_llava_output(llava_image, query)
        return llava_output

    def _check_duplicate(self, image):
        # Check a perceptual hash
        phash = imagehash.phash(image)
        if phash in self.hashes:
            is_duplicate = True
        else:
            is_duplicate = False
        self.hashes[phash] = True
        return is_duplicate

    def filter_data(self):
        ctr = 0
        found_images = 0
        length = len(self.dataset)
        print('Total dataset length', length)
        with MDSWriter(out=self.output, columns=self.fields) as out:
            for sample_id in range(self.start, self.end + 1):
                sample = self.dataset[sample_id]
                ctr += 1
                if ctr % 1000 == 0:
                    print(f'Processed {ctr} of {length} samples. Found {found_images} images.')
                image = sample['image']
                prompt = sample['captions']
                # Untokenize the prompt
                prompt = self.dataset.tokenizer.batch_decode([prompt], skip_special_tokens=True)[0]  # type: ignore
                # Check if it's a duplicate
                if self._check_duplicate(image):
                    continue
                # Compute the score
                score = self.get_aesthetics_score(image)
                # Filter out low scores
                if score <= self.aesthetics_threshold:
                    continue
                # Send it off to llava to tell if it's a photo.
                is_photo = self._llava_classify_photo(image)
                if not is_photo:
                    continue
                # Check for watermarks
                if self._llava_classify_watermark(image):
                    continue

                print('-' * 80)
                print(ctr, prompt, score)
                print('')
                mds_sample = {'image': image, 'captions': prompt, 'aesthetics_score': score}
                if self.generate_caption:
                    llava_caption = self._llava_caption(image, alt_text=prompt)
                    print(llava_caption)
                    mds_sample['llava_caption'] = llava_caption
                print('-' * 80)
                out.write(mds_sample)
                found_images += 1


# Filter dataset
filter = DatasetFilter(args.remotes,
                       args.locals,
                       args.output,
                       image_key=args.image_key,
                       caption_key=args.caption_key,
                       aesthetics_threshold=args.aesthetics_threshold,
                       generate_caption=args.generate_caption,
                       max_size=args.max_size,
                       start=args.start,
                       end=args.end)
filter.filter_data()
