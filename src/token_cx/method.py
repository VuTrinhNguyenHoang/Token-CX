from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

from .banks import BankRepository, infer_activation
from .models import ModelBundle, as_logits, read_config


def minmax(
    tensor: torch.Tensor,
    dims=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    dims = (
        tuple(range(tensor.ndim))
        if dims is None
        else dims
    )

    low = tensor.amin(
        dim=dims,
        keepdim=True,
    )
    high = tensor.amax(
        dim=dims,
        keepdim=True,
    )

    return (tensor - low) / (high - low + eps)


def maxnorm(
    tensor: torch.Tensor,
    dims,
    eps: float = 1e-8,
) -> torch.Tensor:
    maximum = tensor.amax(
        dim=dims,
        keepdim=True,
    )
    return tensor / (maximum + eps)


@dataclass
class Evidence:
    targets: torch.Tensor
    logits: torch.Tensor
    matrix: torch.Tensor
    spatial_weights: torch.Tensor


@dataclass
class Explanation:
    target_class: int
    target_score: float
    activations: torch.Tensor
    relevance: torch.Tensor
    ranking: torch.Tensor
    concept_masks: torch.Tensor
    saliency: torch.Tensor
    exemplars: dict[int, pd.DataFrame]
    diagnostics: dict


def _targets(
    logits: torch.Tensor,
    target,
) -> torch.Tensor:
    if target is None:
        return logits.argmax(1)

    targets = torch.as_tensor(
        target,
        device=logits.device,
        dtype=torch.long,
    ).flatten()

    if len(targets) == 1:
        targets = targets.repeat(len(logits))

    if len(targets) != len(logits):
        raise ValueError(
            "target count must match batch size"
        )

    return targets


def extract_evidence(
    model: ModelBundle,
    images: torch.Tensor,
    target=None,
    token_gradient: bool = True,
    spatial_weighting: bool = True,
) -> Evidence:
    method = model.settings["method"]

    evidence_block = int(
        method["evidence_block"]
    )
    spatial_block = int(
        method["spatial_block"]
    )

    token_module = model.network.blocks[
        evidence_block - 1
    ]
    spatial_module = getattr(
        model.network.blocks[spatial_block - 1],
        method["spatial_target"],
    )

    state = {}
    gradients = {}
    handles = []

    def capture(name, needs_gradient):
        def hook(_, __, output):
            state[name] = output

            if needs_gradient:
                output.register_hook(
                    lambda value: gradients.__setitem__(
                        name,
                        value,
                    )
                )

        return hook

    handles.append(
        token_module.register_forward_hook(
            capture(
                "token",
                token_gradient,
            )
        )
    )

    handles.append(
        spatial_module.register_forward_hook(
            capture(
                "spatial",
                spatial_weighting,
            )
        )
    )

    model.network.zero_grad(set_to_none=True)

    images = images.detach().requires_grad_(
        token_gradient or spatial_weighting
    )

    try:
        with torch.enable_grad():
            logits = as_logits(
                model.network(images)
            )

            targets = _targets(
                logits,
                target,
            )

            if token_gradient or spatial_weighting:
                target_logits = logits.gather(
                    1,
                    targets[:, None],
                )
                target_logits.sum().backward()
    finally:
        for handle in handles:
            handle.remove()

    prefix_tokens = model.num_prefix_tokens

    tokens = state["token"][
        :,
        prefix_tokens:,
    ]

    if token_gradient:
        token_gradients = gradients["token"][
            :,
            prefix_tokens:,
        ]
        tokens = tokens * token_gradients

    token_evidence = torch.relu(tokens)

    if spatial_weighting:
        spatial_activations = state["spatial"][
            :,
            prefix_tokens:,
        ]
        spatial_gradients = gradients["spatial"][
            :,
            prefix_tokens:,
        ]

        channel_weights = spatial_gradients.mean(
            dim=1,
            keepdim=True,
        )

        spatial_weights = torch.relu(
            (
                spatial_activations
                * channel_weights
            ).sum(dim=-1)
        )

        spatial_weights = minmax(
            spatial_weights,
            dims=1,
        )
    else:
        spatial_weights = torch.ones(
            token_evidence.shape[:2],
            dtype=token_evidence.dtype,
            device=token_evidence.device,
        )

    matrix = token_evidence * spatial_weights[..., None]

    matrix = maxnorm(
        matrix,
        dims=(1, 2),
    )

    model.network.zero_grad(set_to_none=True)

    return Evidence(
        targets=targets.detach(),
        logits=logits.detach(),
        matrix=matrix.detach(),
        spatial_weights=spatial_weights.detach(),
    )


def concept_masks(
    activations: torch.Tensor,
    image_size: tuple[int, int],
    gamma: float = 0.5,
) -> torch.Tensor:
    patch_count = activations.shape[0]
    grid_size = int(patch_count**0.5)

    if grid_size * grid_size != patch_count:
        raise ValueError(
            "patch activations must form a square grid"
        )

    masks = activations.T.reshape(
        -1,
        grid_size,
        grid_size,
    )

    masks = minmax(
        masks,
        dims=(1, 2),
    ).pow(gamma)

    masks = F.interpolate(
        masks[:, None],
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )

    return masks[:, 0]


class TokenCX:
    def __init__(
        self,
        model: ModelBundle,
        banks: BankRepository,
        config_path="configs/token_cx.yaml",
    ):
        if model.backbone != banks.backbone:
            raise ValueError(
                "model and concept banks use different backbones"
            )

        method = read_config(config_path)["method"]

        self.model = model
        self.banks = banks
        self.gamma = float(method["mask_gamma"])
        self.inference_iterations = int(
            method["inference_iterations"]
        )

    def explain(
        self,
        image,
        target=None,
        token_gradient: bool = True,
        spatial_weighting: bool = True,
    ) -> Explanation:
        images = self.model.prepare(image)

        if len(images) != 1:
            raise ValueError(
                "explain expects exactly one image"
            )

        evidence = extract_evidence(
            model=self.model,
            images=images,
            target=target,
            token_gradient=token_gradient,
            spatial_weighting=spatial_weighting,
        )

        target_class = int(
            evidence.targets[0]
        )

        bank = self.banks.get(
            target_class
        )

        activations, diagnostics = infer_activation(
            matrix=evidence.matrix[0],
            basis=bank.basis.to(
                evidence.matrix.device
            ),
            max_iterations=self.inference_iterations,
        )

        device_activations = activations.to(
            images.device
        )

        masks = concept_masks(
            activations=device_activations,
            image_size=tuple(images.shape[-2:]),
            gamma=self.gamma,
        )

        relevance = device_activations.max(
            dim=0
        ).values

        saliency = (
            masks
            * relevance[:, None, None]
        ).sum(dim=0)

        saliency = minmax(
            saliency,
            dims=(0, 1),
        )

        ranking = relevance.argsort(
            descending=True
        )

        exemplars = {
            int(concept): bank.examples(
                int(concept)
            )
            for concept in ranking.cpu().tolist()
        }

        target_score = evidence.logits.softmax(
            dim=1
        )[0, target_class]

        return Explanation(
            target_class=target_class,
            target_score=float(target_score.item()),
            activations=activations,
            relevance=relevance.cpu(),
            ranking=ranking.cpu(),
            concept_masks=masks.cpu(),
            saliency=saliency.cpu(),
            exemplars=exemplars,
            diagnostics=diagnostics,
        )
