# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Inference endpoint for LLaVA models."""

import base64
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import snapshot_download
from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX  # type: ignore
from llava.conversation import SeparatorStyle, conv_templates  # type: ignore
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # type: ignore
from llava.model.builder import load_pretrained_model  # type: ignore
from llava.utils import disable_torch_init  # type: ignore
from PIL import Image
from torchvision.transforms import transforms

LOCAL_CHECKPOINT_DIR = '/tmp/llava'


def download_checkpoint(repo_id: str = 'liuhaotian/llava-v1.5-13b',
                        local_checkpoint_dir: str = LOCAL_CHECKPOINT_DIR) -> None:
    """Downloads the checkpoint to the local directory."""
    # Make the local checkpoint directory if it doesn't exist
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    # Download the checkpoint
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_checkpoint_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


class LLaVAInference():
    """LLaVA Inference class."""

    def __init__(self,
                 local_checkpoint_dir: str = LOCAL_CHECKPOINT_DIR,
                 model_base: Optional[str] = None,
                 conv_mode: str = 'llava_v1',
                 temperature: float = 0.2,
                 top_p: Optional[float] = None,
                 num_beams: int = 1,
                 max_new_tokens: int = 1024):
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.device = torch.cuda.current_device()

        disable_torch_init()
        model_path = os.path.expanduser(local_checkpoint_dir)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name)
        self.to_tensor = transforms.ToTensor()

    def _add_image_tokens(self, prompt: str) -> str:
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        return prompt

    def _tokenize(self, prompt: str) -> torch.Tensor:
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids.unsqueeze(0).to(self.device)

    def predict(self, model_requests: List[Dict[str, Any]]):
        results = []
        for req in model_requests:
            if 'input' not in req:
                raise RuntimeError('"input" must be provided to generate call')
            inputs = req['input']
            # Make the tokenized input
            prompt = self._add_image_tokens(inputs['prompt'])
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = self._tokenize(prompt)
            # Make the stopping criteria
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            # Get the image from the bytes and prep it for input
            img_data = base64.b64decode(inputs['image'])
            img = Image.open(BytesIO(img_data))
            img = self.to_tensor(img).unsqueeze(0).to(self.device)  # In range (0, 1)
            image_tensor = self.image_processor.preprocess(img, do_rescale=False, return_tensors='pt')['pixel_values']
            # Forward through the model
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids,
                                                 images=image_tensor.half().cuda(),
                                                 do_sample=True if self.temperature > 0 else False,
                                                 temperature=self.temperature,
                                                 top_p=self.top_p,
                                                 num_beams=self.num_beams,
                                                 max_new_tokens=self.max_new_tokens,
                                                 use_cache=True)
            # Postprocess outputs
            input_token_len = input_ids.shape[1]
            output_text = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            output_text = output_text.strip()
            if output_text.endswith(stop_str):
                output_text = output_text[:-len(stop_str)]
            output_text = output_text.strip()
            results.append(output_text)
        return results
