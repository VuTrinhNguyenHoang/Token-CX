from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from token_cx.method import Explanation, concept_masks
from token_cx.visualization import (
    _select_concepts,
    aligned_patch_crop,
)


def explanation() -> Explanation:
    masks = torch.stack(
        [torch.full((8, 8), value / 7) for value in range(8)]
    )
    return Explanation(
        target_class=100,
        target_score=0.9,
        activations=torch.ones(196, 8),
        relevance=torch.arange(8, dtype=torch.float32),
        ranking=torch.tensor([7, 6, 5, 4, 3, 2, 1, 0]),
        concept_masks=masks,
        saliency=torch.ones(8, 8),
        exemplars={index: pd.DataFrame() for index in range(8)},
        diagnostics={"converged": True},
    )


def test_concept_masks_shape_and_range():
    activation = torch.arange(196 * 8, dtype=torch.float32).reshape(196, 8)
    masks = concept_masks(activation, (224, 224), gamma=0.5)

    assert masks.shape == (8, 224, 224)
    assert float(masks.min()) >= 0
    assert float(masks.max()) <= 1


def test_concept_ids_are_zero_based():
    result = explanation()

    assert _select_concepts(result, [4, 6, 2], 3) == [4, 6, 2]

    with pytest.raises(ValueError):
        _select_concepts(result, [5, 7, 8], 3)


def test_aligned_crop_uses_native_patch_window(tmp_path: Path):
    array = np.zeros((140, 280, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(280, dtype=np.uint8)[None]
    image_path = tmp_path / "train" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(array).save(image_path)

    crop = aligned_patch_crop(
        {"rel_path": "train/sample.png", "patch_id": 15},
        tmp_path,
        image_size=224,
        grid_size=14,
        radius=1,
    )

    expected = Image.open(image_path).convert("RGB").crop(
        (0, 0, 60, 30)
    ).resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    assert crop.size == (224, 224)
    assert np.array_equal(np.asarray(crop), np.asarray(expected))
