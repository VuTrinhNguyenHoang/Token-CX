from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .method import Explanation


def _load_image(value) -> Image.Image:
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")

    return value.convert("RGB")


def overlay(
    image,
    mask,
    alpha: float = 0.5,
    colormap: str = "jet",
):
    mask = torch.as_tensor(
        mask,
        dtype=torch.float32,
    ).detach().cpu().clamp(0, 1).numpy()

    height, width = mask.shape
    image = _load_image(image).resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )

    base = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0
    color = plt.colormaps[colormap](
        mask
    )[..., :3]
    opacity = (
        alpha * mask
    )[..., None]

    return np.clip(
        base * (1 - opacity)
        + color * opacity,
        0,
        1,
    )


def aligned_patch_crop(
    record,
    imagenet_root,
    image_size: int = 224,
    grid_size: int = 14,
    radius: int = 1,
) -> Image.Image:
    """Crop a patch neighborhood on the native image before resizing."""
    image_path = (
        Path(imagenet_root)
        / record["rel_path"]
    )
    image = _load_image(image_path)

    row, column = divmod(
        int(record["patch_id"]),
        grid_size,
    )
    row_start = max(0, row - radius)
    column_start = max(0, column - radius)
    row_stop = min(
        grid_size,
        row + radius + 1,
    )
    column_stop = min(
        grid_size,
        column + radius + 1,
    )

    width, height = image.size
    crop_box = (
        round(column_start * width / grid_size),
        round(row_start * height / grid_size),
        round(column_stop * width / grid_size),
        round(row_stop * height / grid_size),
    )

    return image.crop(crop_box).resize(
        (image_size, image_size),
        Image.Resampling.LANCZOS,
    )


def _select_concepts(
    explanation: Explanation,
    concept_ids,
    top_concepts: int,
) -> list[int]:
    if concept_ids is None:
        selected = explanation.ranking[
            :top_concepts
        ].tolist()
    else:
        selected = [
            int(concept)
            for concept in concept_ids
        ]

    concept_count = len(
        explanation.concept_masks
    )

    if len(selected) != top_concepts:
        raise ValueError(
            f"expected {top_concepts} concept IDs"
        )

    if len(set(selected)) != len(selected):
        raise ValueError(
            "concept IDs must be unique"
        )

    if any(
        concept < 0
        or concept >= concept_count
        for concept in selected
    ):
        raise ValueError(
            "concept IDs must be zero-based values from 0 to K - 1"
        )

    return selected


def plot_saliency(
    image,
    explanation: Explanation,
    output_path=None,
):
    """Plot the query image and activation-weighted Token-CX saliency."""
    height, width = explanation.saliency.shape
    query_image = _load_image(image).resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6.4, 3.1),
        layout="constrained",
    )
    axes[0].imshow(query_image)
    axes[0].set_title("Input")
    axes[1].imshow(
        overlay(
            query_image,
            explanation.saliency,
        )
    )
    axes[1].set_title("Token-CX")

    for axis in axes:
        axis.axis("off")

    _save_figure(figure, output_path)
    return figure


def plot_explanation(
    image,
    explanation: Explanation,
    imagenet_root,
    concept_ids=None,
    top_concepts: int = 3,
    exemplars_per_concept: int = 3,
    output_path=None,
):
    """Render the paper's query-mask and class-bank exemplar layout."""
    if top_concepts != 3:
        raise ValueError(
            "the paper layout requires exactly three concepts"
        )

    if exemplars_per_concept != 3:
        raise ValueError(
            "the paper layout requires exactly three exemplars per concept"
        )

    selected = _select_concepts(
        explanation,
        concept_ids,
        top_concepts,
    )

    height, width = explanation.saliency.shape
    query_image = _load_image(image).resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )

    figure = plt.figure(
        figsize=(10.8, 10.8),
        layout="constrained",
    )
    subfigures = figure.subfigures(
        4,
        1,
        height_ratios=[1, 1, 1, 1],
    )

    query_axes = subfigures[0].subplots(
        1,
        top_concepts,
    )
    subfigures[0].suptitle(
        "(a) Query-specific masks",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )

    for axis, concept in zip(
        query_axes,
        selected,
    ):
        axis.imshow(
            overlay(
                query_image,
                explanation.concept_masks[concept],
            )
        )
        axis.axis("off")

    for position, concept in enumerate(selected):
        panel = chr(ord("b") + position)
        row_axes = subfigures[
            position + 1
        ].subplots(
            1,
            1 + exemplars_per_concept,
        )
        subfigures[position + 1].suptitle(
            f"({panel}) Concept {concept}",
            x=0.01,
            ha="left",
            fontsize=11,
            fontweight="semibold",
        )

        row_axes[0].imshow(
            overlay(
                query_image,
                explanation.concept_masks[concept],
            )
        )
        row_axes[0].axis("off")

        records = explanation.exemplars[
            concept
        ].head(
            exemplars_per_concept
        )

        if len(records) != exemplars_per_concept:
            raise ValueError(
                f"concept {concept} has {len(records)} exemplars; "
                f"expected {exemplars_per_concept}"
            )

        for axis, (_, record) in zip(
            row_axes[1:],
            records.iterrows(),
        ):
            axis.imshow(
                aligned_patch_crop(
                    record,
                    imagenet_root,
                )
            )
            axis.axis("off")

    _save_figure(figure, output_path)
    return figure


def _save_figure(figure, output_path):
    if output_path is None:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=260,
        bbox_inches="tight",
        facecolor="white",
    )

    if output_path.suffix.lower() != ".pdf":
        figure.savefig(
            output_path.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
        )
