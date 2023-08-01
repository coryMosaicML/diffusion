# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Generic gaussian diffusion model in composer."""

from typing import List, Optional

import torch
from composer.models import ComposerModel
from torchmetrics import MeanSquaredError, Metric


class GaussianDiffusion(ComposerModel):