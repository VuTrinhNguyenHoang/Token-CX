from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from PIL import Image
from timm import create_model
from torchvision import transforms


def read_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def as_logits(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return torch.stack(list(output)).mean(0)
    return output


@dataclass
class ModelBundle:
    backbone: str
    network: torch.nn.Module
    preprocess: object
    settings: dict
    device: torch.device

    @property
    def num_prefix_tokens(self) -> int:
        return int(getattr(self.network, "num_prefix_tokens", 1))

    def prepare(self, images) -> torch.Tensor:
        if torch.is_tensor(images):
            batch = images.unsqueeze(0) if images.ndim == 3 else images
            return batch.to(self.device)

        images = images if isinstance(images, (list, tuple)) else [images]
        batch = []

        for image in images:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            batch.append(self.preprocess(image))

        return torch.stack(batch).to(self.device)


def load_model(
    backbone: str,
    config_path: str | Path = "configs/token_cx.yaml",
    device: str | torch.device | None = None,
) -> ModelBundle:
    config = read_config(config_path)

    if backbone not in config["models"]:
        raise ValueError(f"Unknown backbone: {backbone}")

    settings = config["models"][backbone]
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    network = create_model(
        settings["timm_name"],
        pretrained=True,
    )
    network = network.to(device).eval().requires_grad_(False)

    pretrained = network.pretrained_cfg
    image_size = int(settings["image_size"])

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                pretrained["mean"],
                pretrained["std"],
            ),
        ]
    )

    return ModelBundle(
        backbone=backbone,
        network=network,
        preprocess=preprocess,
        settings=config,
        device=device,
    )
