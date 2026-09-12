"""Model definition and preprocessing for the AI image detector.

This is the single source of truth for the network architecture. Both the
training scripts and the web app import ``AIImageDetector`` from here so the
architecture can never drift out of sync with the saved weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Index 0 -> AI generated, index 1 -> human made. This order matches
# torchvision.datasets.ImageFolder, which sorts class folders alphabetically
# ("Fake" before "Real").
CLASS_NAMES: tuple[str, str] = ("AI generated", "Human made")

IMAGE_SIZE: int = 250


class AIImageDetector(nn.Module):
    """A compact CNN: two conv/pool blocks followed by five fully connected layers.

    Input:  RGB tensor of shape (N, 3, 250, 250)
    Output: raw logits of shape (N, 2)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 59 * 59, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 59 * 59)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def build_transform() -> transforms.Compose:
    """Return the exact preprocessing pipeline used during training."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


# The original notebooks named the conv layers ``con1``/``con2``. The released
# checkpoint was saved with those keys, so remap them to the current names when
# loading rather than carrying the typo forward in the code.
_LEGACY_KEY_MAP = {
    "con1.weight": "conv1.weight",
    "con1.bias": "conv1.bias",
    "con2.weight": "conv2.weight",
    "con2.bias": "conv2.bias",
}


def _normalize_state_dict(state: dict) -> dict:
    return {_LEGACY_KEY_MAP.get(key, key): value for key, value in state.items()}


def load_model(weights_path: str, device: str | torch.device = "cpu") -> AIImageDetector:
    """Instantiate the network and load weights from ``weights_path`` in eval mode."""
    model = AIImageDetector()
    state = torch.load(weights_path, map_location=torch.device(device))
    model.load_state_dict(_normalize_state_dict(state))
    model.to(device)
    model.eval()
    return model
