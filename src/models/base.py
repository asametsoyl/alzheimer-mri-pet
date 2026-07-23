"""Abstract base class for all synthesis models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class BaseModel(ABC, nn.Module):
    """Abstract base for all MRI→PET synthesis models.

    All concrete models must implement:
    - forward(mri) -> synthesized_pet
    - get_config() -> serializable dict describing the model
    """

    @abstractmethod
    def forward(self, mri: Tensor) -> Tensor:
        """Synthesize PET from MRI input.

        Args:
            mri: MRI tensor of shape (B, 1, D, H, W).

        Returns:
            Synthesized PET tensor of shape (B, 1, D, H, W).
        """

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            Count of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def get_config(self) -> dict:
        """Return a serializable dict describing model architecture.

        Returns:
            Config dictionary suitable for YAML/JSON serialization.
        """
