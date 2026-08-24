from __future__ import annotations

import torch
import torch.nn.functional as F

from .method import minmax
from .models import as_logits


def auc(values) -> float:
    values = torch.as_tensor(
        values,
        dtype=torch.float32,
    )

    if len(values) < 2:
        raise ValueError(
            "AUC requires at least two points"
        )

    return float(
        torch.trapezoid(
            values,
            dx=1.0 / (len(values) - 1),
        )
    )


@torch.no_grad()
def perturbation_curve(
    model,
    image: torch.Tensor,
    saliency: torch.Tensor,
    target: int,
    mode: str,
    steps: int = 100,
    batch_size: int = 64,
):
    network = (
        model.network
        if hasattr(model, "network")
        else model
    )

    _, _, height, width = image.shape
    saliency = saliency.to(image.device)

    if saliency.shape != (height, width):
        saliency = F.interpolate(
            saliency[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    order = minmax(
        saliency,
        dims=(0, 1),
    ).flatten().argsort(
        descending=True
    )

    counts = torch.linspace(
        0,
        height * width,
        steps + 1,
        device=image.device,
    )
    counts = counts.round().long()

    scores = []

    for start in range(
        0,
        len(counts),
        batch_size,
    ):
        masks = []

        for count in counts[
            start : start + batch_size
        ]:
            mask = torch.zeros(
                height * width,
                device=image.device,
            )

            mask[order[: int(count)]] = 1

            masks.append(
                mask.reshape(
                    1,
                    height,
                    width,
                )
            )

        masks = torch.stack(masks)

        inputs = image.expand(
            len(masks),
            -1,
            -1,
            -1,
        )

        baseline = torch.zeros_like(inputs)

        if mode == "deletion":
            perturbed = (
                inputs * (1 - masks)
                + baseline * masks
            )
        elif mode == "insertion":
            perturbed = (
                baseline * (1 - masks)
                + inputs * masks
            )
        else:
            raise ValueError(
                "mode must be 'deletion' or 'insertion'"
            )

        probabilities = as_logits(
            network(perturbed)
        ).softmax(dim=1)[:, int(target)]

        scores.extend(
            probabilities.cpu().tolist()
        )

    return {
        "auc": auc(scores),
        "curve": scores,
        "counts": counts.cpu().tolist(),
    }


@torch.no_grad()
def perturbation_curves(
    model,
    image: torch.Tensor,
    saliencies: torch.Tensor,
    target: int,
    steps: int = 100,
    batch_size: int = 128,
):
    """Evaluate deletion and insertion in the paper's shared batches."""
    network = (
        model.network
        if hasattr(model, "network")
        else model
    )

    _, _, height, width = image.shape
    saliencies = torch.as_tensor(
        saliencies,
        dtype=image.dtype,
        device=image.device,
    )

    if saliencies.ndim == 2:
        saliencies = saliencies[None]

    if saliencies.shape[-2:] != (height, width):
        saliencies = F.interpolate(
            saliencies[:, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    candidate_count = len(saliencies)
    pixel_count = height * width
    step_count = steps + 1

    order = minmax(
        saliencies,
        dims=(1, 2),
    ).flatten(1).argsort(
        dim=1,
        descending=True,
    )

    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(
            pixel_count,
            device=image.device,
        ).expand_as(order),
    )

    counts = torch.linspace(
        0,
        pixel_count,
        step_count,
        device=image.device,
    ).round().long()

    task_count = candidate_count * 2 * step_count
    task_scores = torch.empty(
        task_count,
        device=image.device,
    )

    for start in range(0, task_count, batch_size):
        stop = min(start + batch_size, task_count)
        task_ids = torch.arange(
            start,
            stop,
            device=image.device,
        )

        candidate_ids = task_ids // (2 * step_count)
        mode_ids = (task_ids // step_count) % 2
        step_ids = task_ids % step_count

        masks = (
            ranks[candidate_ids]
            < counts[step_ids, None]
        ).reshape(
            -1,
            1,
            height,
            width,
        ).to(image.dtype)

        inputs = image.expand(
            len(task_ids),
            -1,
            -1,
            -1,
        )

        insertion = inputs * masks
        perturbed = torch.where(
            mode_ids[:, None, None, None].bool(),
            insertion,
            inputs - insertion,
        )

        task_scores[start:stop] = as_logits(
            network(perturbed)
        ).softmax(dim=1)[:, int(target)]

    curves = task_scores.reshape(
        candidate_count,
        2,
        step_count,
    ).cpu()

    return {
        "deletion_auc": torch.trapezoid(
            curves[:, 0],
            dx=1.0 / steps,
            dim=1,
        ).tolist(),
        "insertion_auc": torch.trapezoid(
            curves[:, 1],
            dx=1.0 / steps,
            dim=1,
        ).tolist(),
        "deletion_curves": curves[:, 0].tolist(),
        "insertion_curves": curves[:, 1].tolist(),
        "counts": counts.cpu().tolist(),
    }


def evaluate_saliency(
    model,
    image,
    saliency,
    target,
    steps=100,
    batch_size=128,
):
    result = perturbation_curves(
        model=model,
        image=image,
        saliencies=saliency,
        target=target,
        steps=steps,
        batch_size=batch_size,
    )

    return {
        "deletion_auc": result["deletion_auc"][0],
        "insertion_auc": result["insertion_auc"][0],
        "deletion_curve": result["deletion_curves"][0],
        "insertion_curve": result["insertion_curves"][0],
        "counts": result["counts"],
    }


@torch.inference_mode()
def concept_intervention_scores(
    model,
    image: torch.Tensor,
    masks: torch.Tensor,
    target: int,
    batch_size: int = 12,
):
    """Score masks for qualitative ordering only, not saliency aggregation."""
    network = (
        model.network
        if hasattr(model, "network")
        else model
    )

    if image.ndim != 4 or len(image) != 1:
        raise ValueError(
            "image must have shape [1, channels, height, width]"
        )

    masks = torch.as_tensor(
        masks,
        dtype=image.dtype,
        device=image.device,
    )

    if (
        masks.ndim != 3
        or tuple(masks.shape[-2:])
        != tuple(image.shape[-2:])
    ):
        raise ValueError(
            "masks must have shape [concepts, height, width]"
        )

    baseline = torch.zeros_like(image)
    original_logit = as_logits(
        network(image)
    )[:, int(target)]
    baseline_logit = as_logits(
        network(baseline)
    )[:, int(target)]

    keep_logits = []
    drop_logits = []

    for start in range(0, len(masks), batch_size):
        mask = masks[
            start : start + batch_size,
            None,
        ]
        inputs = image.expand(
            len(mask),
            -1,
            -1,
            -1,
        )

        keep_logits.append(
            as_logits(network(inputs * mask))[
                :,
                int(target),
            ]
        )
        drop_logits.append(
            as_logits(network(inputs * (1 - mask)))[
                :,
                int(target),
            ]
        )

    keep_logits = torch.cat(keep_logits)
    drop_logits = torch.cat(drop_logits)

    keep = torch.relu(
        keep_logits - baseline_logit
    )
    drop = torch.relu(
        original_logit - drop_logits
    )
    keep_drop = torch.relu(
        (keep_logits - baseline_logit)
        + (original_logit - drop_logits)
    )

    return {
        "keep": keep.cpu(),
        "drop": drop.cpu(),
        "keep_drop": keep_drop.cpu(),
        "keep_logits": keep_logits.cpu(),
        "drop_logits": drop_logits.cpu(),
        "original_logit": float(
            original_logit.item()
        ),
        "baseline_logit": float(
            baseline_logit.item()
        ),
    }


def pointing_game(
    saliency,
    boxes,
    original_size,
) -> int:
    height, width = saliency.shape
    original_width, original_height = original_size

    index = int(saliency.argmax())

    x = (
        index % width
    ) * original_width / width

    y = (
        index // width
    ) * original_height / height

    return int(
        any(
            xmin <= x <= xmax
            and ymin <= y <= ymax
            for xmin, ymin, xmax, ymax in boxes
        )
    )
