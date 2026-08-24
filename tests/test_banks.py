from pathlib import Path

import pandas as pd
import torch

from token_cx.banks import (
    BankRepository,
    save_bank_artifacts,
    select_exemplars,
)


CONFIG = Path(__file__).parents[1] / "configs/token_cx.yaml"


def test_select_exemplars_uses_one_patch_per_image():
    activation = torch.tensor(
        [
            [1.0, 7.0],
            [5.0, 2.0],
            [3.0, 4.0],
            [2.0, 8.0],
        ]
    )
    samples = pd.DataFrame(
        [
            {"sample_id": "first", "class_idx": 0},
            {"sample_id": "second", "class_idx": 0},
        ]
    )

    rows = select_exemplars(activation, samples, top_k=2)

    concept_zero = rows[rows["concept_idx"] == 0]
    concept_one = rows[rows["concept_idx"] == 1]
    assert concept_zero["sample_id"].tolist() == ["first", "second"]
    assert concept_zero["patch_id"].tolist() == [1, 0]
    assert concept_one["sample_id"].tolist() == ["second", "first"]
    assert concept_one["patch_id"].tolist() == [1, 0]


def test_local_class_banks_round_trip(tmp_path: Path):
    for class_id in (4, 2):
        exemplars = pd.DataFrame(
            [
                {
                    "class_idx": class_id,
                    "concept_idx": 0,
                    "exemplar_rank": 1,
                    "sample_id": f"sample-{class_id}",
                }
            ]
        )
        save_bank_artifacts(
            tmp_path / f"class_{class_id:04d}",
            bases=torch.full((1, 4, 8), float(class_id)),
            class_indices=torch.tensor([class_id]),
            exemplars=exemplars,
            metadata={"rank": 8, "block_number": 9},
            diagnostics=pd.DataFrame([{"class_idx": class_id}]),
        )

    banks = BankRepository.from_directory("deit", tmp_path, CONFIG)

    assert banks.class_indices.tolist() == [2, 4]
    assert banks.bases.shape == (2, 4, 8)
    assert banks.get(4).examples(0)["sample_id"].tolist() == ["sample-4"]
