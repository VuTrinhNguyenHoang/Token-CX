from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

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


def _error_terms(matrix, activation, basis) -> tuple[float, float]:
    residual = matrix - activation @ basis.T
    return float(residual.square().sum()), float(matrix.square().sum())


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

    reconstruction_error, input_norm = _error_terms(
        matrix,
        activation,
        basis,
    )
    diagnostics = {
        "relative_error": _relative_error(
            matrix,
            activation,
            basis,
        ),
        "reconstruction_squared_error": reconstruction_error,
        "input_squared_norm": input_norm,
        "initial_relative_error": initial_error,
        "nmf_epochs": int(epochs),
        "nmf_steps": int(epochs),
        "nmf_solver": "torch_mu",
        "nmf_init": "nndsvda",
        "nmf_init_iters": int(init_iterations),
        "nmf_device": str(matrix.device),
        "basis_normalization": "l2_columns",
        "basis_scale_min_before": float(scales.min()),
        "basis_scale_max_before": float(scales.max()),
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

    reconstruction_error, input_norm = _error_terms(
        matrix,
        activation,
        basis,
    )
    diagnostics = {
        "relative_error": _relative_error(
            matrix,
            activation,
            basis,
        ),
        "reconstruction_squared_error": reconstruction_error,
        "input_squared_norm": input_norm,
        "relative_change": float(change),
        "iterations": int(iteration),
        "converged": converged,
    }

    return activation.cpu(), diagnostics


def select_exemplars(
    activation,
    samples,
    top_k: int = 10,
) -> pd.DataFrame:
    """Select one strongest patch per image, then rank images per concept."""
    activation = _float_tensor(activation).cpu()
    records = (
        samples.to_dict("records")
        if isinstance(samples, pd.DataFrame)
        else list(samples)
    )

    if not records or top_k < 1:
        raise ValueError(
            "samples must be non-empty and top_k must be positive"
        )

    image_count = len(records)
    if activation.ndim != 2 or len(activation) % image_count:
        raise ValueError(
            "activation rows must divide evenly across samples"
        )

    patches_per_image = len(activation) // image_count
    activation = activation.reshape(
        image_count,
        patches_per_image,
        -1,
    )
    values, patch_ids = activation.max(dim=1)

    rows = []
    for concept_id in range(activation.shape[-1]):
        scores, image_ids = values[:, concept_id].topk(
            min(top_k, image_count)
        )
        for exemplar_rank, (score, image_id) in enumerate(
            zip(scores, image_ids),
            start=1,
        ):
            image_id = int(image_id)
            row = dict(records[image_id])
            row.update(
                {
                    "concept_idx": int(concept_id),
                    "exemplar_rank": int(exemplar_rank),
                    "patch_id": int(
                        patch_ids[image_id, concept_id]
                    ),
                    "activation": float(score),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


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
        diagnostics=None,
    ):
        self.backbone = backbone
        self.bases = bases.float().cpu()
        self.class_indices = class_indices.long().cpu()
        self.exemplars = exemplars
        self.metadata = metadata
        self.diagnostics = diagnostics

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
            "diagnostics.parquet",
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
            diagnostics=pd.read_parquet(
                paths["diagnostics.parquet"]
            ),
        )

    @classmethod
    def from_directory(
        cls,
        backbone: str,
        directory: str | Path,
        config_path: str | Path = "configs/token_cx.yaml",
    ):
        """Load one aggregate bank or a directory of completed class banks."""
        directory = Path(directory)
        config = read_config(config_path)

        if backbone not in config["models"]:
            raise ValueError(
                f"Unknown backbone: {backbone}"
            )

        if (directory / "bases.safetensors").exists():
            artifact_directories = [directory]
        else:
            artifact_directories = sorted(
                path.parent
                for path in directory.glob(
                    "class_*/bases.safetensors"
                )
            )

        if not artifact_directories:
            raise FileNotFoundError(
                f"No bank artifacts found in {directory}"
            )

        bases = []
        class_indices = []
        exemplar_frames = []
        metadata_parts = []
        diagnostic_frames = []
        expected_rank = int(config["method"]["rank"])
        expected_block = int(
            config["method"]["evidence_block"]
        )

        for artifact_directory in artifact_directories:
            tensors = load_file(
                artifact_directory / "bases.safetensors"
            )
            part_bases = tensors["bases"].float()
            part_classes = tensors["class_indices"].long()

            if part_bases.ndim == 2:
                part_bases = part_bases[None]

            if len(part_bases) != len(part_classes):
                raise ValueError(
                    f"Invalid bank shapes in {artifact_directory}"
                )

            metadata = json.loads(
                (
                    artifact_directory / "config.json"
                ).read_text(encoding="utf-8")
            )

            if int(metadata["rank"]) != expected_rank:
                raise ValueError(
                    f"Rank mismatch in {artifact_directory}"
                )
            if int(metadata["block_number"]) != expected_block:
                raise ValueError(
                    f"Block mismatch in {artifact_directory}"
                )

            bases.append(part_bases)
            class_indices.append(part_classes)
            exemplar_frames.append(
                pd.read_parquet(
                    artifact_directory / "exemplars.parquet"
                )
            )
            metadata_parts.append(metadata)
            diagnostic_path = (
                artifact_directory / "diagnostics.parquet"
            )
            if diagnostic_path.exists():
                diagnostic_frames.append(
                    pd.read_parquet(diagnostic_path)
                )

        bases = torch.cat(bases)
        class_indices = torch.cat(class_indices)

        if len(class_indices.unique()) != len(class_indices):
            raise ValueError(
                "Local bank directory contains duplicate classes"
            )

        order = class_indices.argsort()
        bases = bases[order]
        class_indices = class_indices[order]
        exemplars = pd.concat(
            exemplar_frames,
            ignore_index=True,
        )
        exemplars = exemplars.sort_values(
            ["class_idx", "concept_idx", "exemplar_rank"]
        ).reset_index(drop=True)

        if diagnostic_frames and len(diagnostic_frames) != len(
            artifact_directories
        ):
            raise ValueError(
                "Local bank diagnostics are incomplete"
            )

        return cls(
            backbone=backbone,
            bases=bases,
            class_indices=class_indices,
            exemplars=exemplars,
            metadata={
                "source": "local",
                "directory": str(directory.resolve()),
                "rank": expected_rank,
                "block_number": expected_block,
                "parts": metadata_parts,
            },
            diagnostics=(
                pd.concat(diagnostic_frames, ignore_index=True)
                if diagnostic_frames
                else None
            ),
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


def save_bank_artifacts(
    directory: str | Path,
    bases,
    class_indices,
    exemplars: pd.DataFrame,
    metadata: dict,
    diagnostics: pd.DataFrame | None = None,
) -> Path:
    """Save banks using the same file schema as the public artifacts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)

    bases = _float_tensor(bases).cpu().contiguous()
    class_indices = torch.as_tensor(
        class_indices,
        dtype=torch.long,
    ).cpu().contiguous()

    if bases.ndim == 2:
        bases = bases[None]

    if bases.ndim != 3 or len(bases) != len(class_indices):
        raise ValueError(
            "bases must be [classes, dimensions, concepts]"
        )

    save_file(
        {
            "bases": bases,
            "class_indices": class_indices,
        },
        directory / "bases.safetensors",
    )
    exemplars.to_parquet(
        directory / "exemplars.parquet",
        index=False,
    )
    (directory / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if diagnostics is not None:
        diagnostics.to_parquet(
            directory / "diagnostics.parquet",
            index=False,
        )

    return directory
