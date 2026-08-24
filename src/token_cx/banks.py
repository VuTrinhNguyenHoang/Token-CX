from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models import read_config


def _float_tensor(value, device=None) -> torch.Tensor:
    default_device = (
        value.device if torch.is_tensor(value) else torch.device("cpu")
    )
    return torch.as_tensor(
        value,
        dtype=torch.float32,
        device=device or default_device,
    )


def _relative_error(matrix, activation, basis) -> float:
    eps = torch.finfo(matrix.dtype).eps
    residual = torch.linalg.vector_norm(
        matrix - activation @ basis.T
    )
    denominator = torch.linalg.vector_norm(matrix) + eps
    return float(residual / denominator)


def normalize_basis(
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    basis = _float_tensor(basis)
    eps = torch.finfo(basis.dtype).eps

    scales = torch.linalg.vector_norm(
        basis,
        dim=0,
    ).clamp_min(eps)

    return basis / scales, scales


def _nndsvda(
    matrix,
    rank,
    random_state=0,
    power_iterations=4,
):
    devices = [matrix.device.index] if matrix.is_cuda else []

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(random_state)

        left, singular_values, right = torch.pca_lowrank(
            matrix,
            q=rank,
            center=False,
            niter=power_iterations,
        )

    eps = torch.finfo(matrix.dtype).eps

    activation = matrix.new_zeros(
        matrix.shape[0],
        rank,
    )
    basis = matrix.new_zeros(
        matrix.shape[1],
        rank,
    )

    activation[:, 0] = (
        singular_values[0].sqrt() * left[:, 0].abs()
    )
    basis[:, 0] = (
        singular_values[0].sqrt() * right[:, 0].abs()
    )

    for index in range(1, rank):
        left_positive = left[:, index].clamp_min(0)
        left_negative = (-left[:, index]).clamp_min(0)
        right_positive = right[:, index].clamp_min(0)
        right_negative = (-right[:, index]).clamp_min(0)

        positive_scale = (
            torch.linalg.vector_norm(left_positive)
            * torch.linalg.vector_norm(right_positive)
        )
        negative_scale = (
            torch.linalg.vector_norm(left_negative)
            * torch.linalg.vector_norm(right_negative)
        )

        if positive_scale >= negative_scale:
            left_vector = left_positive / (
                torch.linalg.vector_norm(left_positive).clamp_min(eps)
            )
            right_vector = right_positive / (
                torch.linalg.vector_norm(right_positive).clamp_min(eps)
            )
            scale = positive_scale
        else:
            left_vector = left_negative / (
                torch.linalg.vector_norm(left_negative).clamp_min(eps)
            )
            right_vector = right_negative / (
                torch.linalg.vector_norm(right_negative).clamp_min(eps)
            )
            scale = negative_scale

        coefficient = (
            singular_values[index] * scale
        ).sqrt()

        activation[:, index] = coefficient * left_vector
        basis[:, index] = coefficient * right_vector

    average = matrix.mean().clamp_min(eps)

    activation = torch.where(
        activation == 0,
        average,
        activation,
    )
    basis = torch.where(
        basis == 0,
        average,
        basis,
    )

    return activation, basis


@torch.no_grad()
def fit_basis(
    matrix,
    rank: int,
    epochs: int = 200,
    random_state: int = 0,
    init_iterations: int = 4,
    device=None,
):
    matrix = _float_tensor(matrix, device)

    if matrix.ndim != 2 or rank > min(matrix.shape):
        raise ValueError(
            "matrix must be 2D and support the requested rank"
        )

    if torch.any(matrix < 0):
        raise ValueError("NMF input must be non-negative")

    activation, basis = _nndsvda(
        matrix,
        rank,
        random_state=random_state,
        power_iterations=init_iterations,
    )

    initial_error = _relative_error(
        matrix,
        activation,
        basis,
    )

    eps = torch.finfo(matrix.dtype).eps

    for _ in range(epochs):
        basis *= (
            matrix.T @ activation
        ) / (
            basis @ (activation.T @ activation) + eps
        )

        activation *= (
            matrix @ basis
        ) / (
            activation @ (basis.T @ basis) + eps
        )

    basis, scales = normalize_basis(basis)
    activation *= scales

    diagnostics = {
        "initial_relative_error": initial_error,
        "relative_error": _relative_error(
            matrix,
            activation,
            basis,
        ),
        "nmf_epochs": int(epochs),
        "nmf_init": "nndsvda",
        "nmf_init_iterations": int(init_iterations),
        "basis_normalization": "l2_columns",
    }

    return (
        activation.cpu(),
        basis.cpu(),
        diagnostics,
    )


@torch.no_grad()
def infer_activation(
    matrix,
    basis,
    max_iterations: int = 300,
    min_iterations: int = 20,
    check_every: int = 10,
    tolerance: float = 1e-4,
):
    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be positive"
        )

    effective_min_iterations = min(
        int(min_iterations),
        int(max_iterations),
    )

    matrix = _float_tensor(matrix)
    basis, _ = normalize_basis(
        _float_tensor(basis, matrix.device)
    )

    if (
        matrix.ndim != 2
        or basis.ndim != 2
        or matrix.shape[1] != basis.shape[0]
    ):
        raise ValueError(
            "matrix and basis shapes are incompatible"
        )

    if torch.any(matrix < 0) or torch.any(basis < 0):
        raise ValueError(
            "fixed-basis NMF requires non-negative inputs"
        )

    eps = torch.finfo(matrix.dtype).eps
    numerator = matrix @ basis
    gram = basis.T @ basis

    activation = numerator / (
        gram.diag().clamp_min(eps)
    )

    change = torch.tensor(
        float("inf"),
        device=matrix.device,
    )
    converged = False

    for iteration in range(1, max_iterations + 1):
        updated = activation * numerator / (
            activation @ gram + eps
        )

        should_check = (
            iteration >= effective_min_iterations
            and (
                iteration % check_every == 0
                or iteration == max_iterations
            )
        )

        if should_check:
            change = torch.linalg.vector_norm(
                updated - activation
            ) / (
                torch.linalg.vector_norm(activation) + eps
            )

        activation = updated

        if should_check and change < tolerance:
            converged = True
            break

    diagnostics = {
        "relative_error": _relative_error(
            matrix,
            activation,
            basis,
        ),
        "relative_change": float(change),
        "iterations": int(iteration),
        "converged": converged,
    }

    return activation.cpu(), diagnostics


@dataclass(frozen=True)
class ConceptBank:
    class_id: int
    basis: torch.Tensor
    exemplars: pd.DataFrame

    def examples(
        self,
        concept_id: int,
        limit: int | None = None,
    ) -> pd.DataFrame:
        rows = self.exemplars[
            self.exemplars["concept_idx"] == concept_id
        ]
        rows = rows.sort_values("exemplar_rank")

        if limit is not None:
            rows = rows.head(limit)

        return rows


class BankRepository:
    def __init__(
        self,
        backbone,
        bases,
        class_indices,
        exemplars,
        metadata,
    ):
        self.backbone = backbone
        self.bases = bases.float().cpu()
        self.class_indices = class_indices.long().cpu()
        self.exemplars = exemplars
        self.metadata = metadata

        self._positions = {
            int(class_id): index
            for index, class_id in enumerate(
                self.class_indices
            )
        }

        if (
            self.bases.ndim != 3
            or len(self.bases) != len(self.class_indices)
        ):
            raise ValueError(
                "invalid class-bank tensor shape"
            )

        if torch.any(self.bases < 0):
            raise ValueError(
                "concept bases must be non-negative"
            )

    @classmethod
    def from_config(
        cls,
        backbone: str,
        config_path: str | Path = "configs/token_cx.yaml",
        artifact_dir: str | Path | None = None,
    ):
        config_path = Path(config_path).resolve()
        config = read_config(config_path)

        if backbone not in config["models"]:
            raise ValueError(
                f"Unknown backbone: {backbone}"
            )

        artifact = config["artifacts"]
        model_config = config["models"][backbone]
        folder = (
            f"banks/{model_config['artifact_folder']}"
        )

        if artifact_dir is None:
            artifact_dir = (
                config_path.parents[1] / ".artifacts"
            )

        artifact_dir = Path(artifact_dir)
        paths = {}

        for name in (
            "bases.safetensors",
            "exemplars.parquet",
            "config.json",
        ):
            paths[name] = hf_hub_download(
                repo_id=artifact["repo_id"],
                filename=f"{folder}/{name}",
                repo_type="dataset",
                revision=artifact["revision"],
                local_dir=artifact_dir,
            )

        tensors = load_file(
            paths["bases.safetensors"]
        )

        metadata = json.loads(
            Path(paths["config.json"]).read_text(
                encoding="utf-8"
            )
        )

        if metadata["rank"] != config["method"]["rank"]:
            raise ValueError(
                "artifact rank does not match token_cx.yaml"
            )

        if (
            metadata["block_number"]
            != config["method"]["evidence_block"]
        ):
            raise ValueError(
                "artifact block does not match token_cx.yaml"
            )

        exemplars = pd.read_parquet(
            paths["exemplars.parquet"]
        )

        return cls(
            backbone=backbone,
            bases=tensors["bases"],
            class_indices=tensors["class_indices"],
            exemplars=exemplars,
            metadata=metadata,
        )

    def get(self, class_id: int) -> ConceptBank:
        if class_id not in self._positions:
            raise KeyError(
                f"No concept bank for class {class_id}"
            )

        position = self._positions[class_id]

        exemplars = self.exemplars[
            self.exemplars["class_idx"] == class_id
        ]

        return ConceptBank(
            class_id=class_id,
            basis=self.bases[position],
            exemplars=exemplars,
        )
