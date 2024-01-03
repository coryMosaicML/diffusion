# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Custom torchmetrics Metric for computing PickScore of two models."""

from typing import List, Union

import torch
from PIL import Image
from torchmetrics import Metric
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModel, AutoProcessor


class PickScoreMetric(Metric):
    """Custom torchmetrics Metric for computing PickScore of two models."""

    def __init__(self):
        super().__init__()
        # Create the PickScore model
        self.pickscore_processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        self.pickscore_model = AutoModel.from_pretrained('yuvalkirstain/PickScore_v1').eval()
        # Storage for the running stats
        self.add_state('total_prob', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_items', default=torch.tensor(0), dist_reduce_fx='sum')

    def _pickscore_image_pair(self, prompt: str, baseline_image, model_image):
        """Function that takes in a prompt and a pair of PIL image and returns the pick score of the second image."""
        # Preprocess images
        image_inputs = self.pickscore_processor(
            images=[baseline_image, model_image],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt',
        ).to(self.device)
        # Preprocess prompts
        text_inputs = self.pickscore_processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            # Embed images
            image_embs = self.pickscore_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            # Embed prompts
            text_embs = self.pickscore_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            # Combine embeddings to get the score
            scores = self.pickscore_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            # Calculate pick probs
            probs = torch.softmax(scores, dim=-1)
        return probs[-1].item()

    def update(self, prompts: List[str], baseline_images: Union[torch.Tensor, List[Image.Image]],
               model_images: Union[torch.Tensor, List[Image.Image]]):
        """Update the PickScore metric with a batch of prompts, baseline images, and model images."""
        for prompt, baseline_image, model_image in zip(prompts, baseline_images, model_images):
            if isinstance(baseline_image, torch.Tensor):
                baseline_image = to_pil_image(baseline_image)
            if isinstance(model_image, torch.Tensor):
                model_image = to_pil_image(model_image)
            pickscore = self._pickscore_image_pair(prompt, baseline_image, model_image)
            self.total_prob += 1.0 if pickscore > 0.5 else 0.0  # type: ignore
            self.total_items += 1  # type: ignore

    def compute(self):
        """Compute the PickScore metric."""
        return self.total_prob.item() / self.total_items.item()  # type: ignore
