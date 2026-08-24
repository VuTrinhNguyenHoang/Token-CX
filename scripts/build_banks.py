from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from token_cx import load_model
from token_cx.banks import (
    fit_basis,
    save_bank_artifacts,
    select_exemplars,
)
from token_cx.method import extract_evidence
from token_cx.models import read_config


MODEL_IDS = {
    "vit": "vit",
    "deit": "deit_distilled",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild Token-CX class-specific concept banks."
    )
    parser.add_argument("--backbone", choices=sorted(MODEL_IDS), required=True)
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/token_cx.yaml"),
    )
    parser.add_argument("--class-start", type=int, default=0)
    parser.add_argument("--class-stop", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(".artifacts"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(values) -> str:
    text = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def identity_hash(identity: dict) -> str:
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fixed_manifest(config, artifact_dir: Path) -> tuple[pd.DataFrame, str]:
    artifact = config["artifacts"]
    common = {
        "repo_id": artifact["repo_id"],
        "repo_type": "dataset",
        "revision": artifact["revision"],
        "local_dir": artifact_dir,
    }
    manifest_path = Path(
        hf_hub_download(
            filename="manifests/bank_samples.csv",
            **common,
        )
    )
    artifact_manifest_path = Path(
        hf_hub_download(
            filename="artifact_manifest.json",
            **common,
        )
    )
    artifact_manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    expected_hash = artifact_manifest["bank_manifest_sha256"]
    actual_hash = sha256(manifest_path)
    if actual_hash != expected_hash:
        raise ValueError("The bank manifest does not match the pinned artifact")

    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "image_id": str, "synset": str},
    )
    return frame, actual_hash


def validate_manifest(frame: pd.DataFrame, images_per_class: int):
    counts = frame.groupby("class_idx", sort=False).size()
    if len(counts) != 1000 or not counts.eq(images_per_class).all():
        raise ValueError(
            "The bank manifest must contain 1000 balanced ImageNet classes"
        )


def collect_class_evidence(
    model,
    frame: pd.DataFrame,
    imagenet_root: Path,
    class_id: int,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    records = frame.to_dict("records")
    progress = tqdm(
        range(0, len(records), batch_size),
        desc=f"class {class_id:04d}: evidence",
        leave=False,
        unit="batch",
    )
    for start in progress:
        batch = records[start : start + batch_size]
        paths = [imagenet_root / record["rel_path"] for record in batch]
        missing = next((path for path in paths if not path.exists()), None)
        if missing is not None:
            raise FileNotFoundError(missing)

        images = model.prepare(paths)
        evidence = extract_evidence(
            model=model,
            images=images,
            target=class_id,
        )
        parts.append(evidence.matrix.cpu())

    return torch.cat(parts)


def build_class(
    model,
    config,
    backbone: str,
    class_id: int,
    samples: pd.DataFrame,
    manifest_hash: str,
    imagenet_root: Path,
    output_root: Path,
    batch_size: int,
):
    final_directory = output_root / f"class_{class_id:04d}"
    required = {
        "bases.safetensors",
        "exemplars.parquet",
        "diagnostics.parquet",
        "config.json",
    }
    if final_directory.exists():
        present = {path.name for path in final_directory.iterdir()}
        if required <= present:
            return "skipped"
        raise RuntimeError(
            f"Incomplete bank directory exists: {final_directory}"
        )

    method = config["method"]
    bank_config = config["bank"]
    evidence = collect_class_evidence(
        model,
        samples,
        imagenet_root,
        class_id,
        batch_size,
    )
    image_count, patch_count, dimensions = evidence.shape
    if patch_count != int(bank_config["patches_per_image"]):
        raise ValueError("Unexpected patch-token count")

    activation, basis, diagnostics = fit_basis(
        evidence.reshape(-1, dimensions),
        rank=int(method["rank"]),
        epochs=int(bank_config["nmf_epochs"]),
        random_state=int(bank_config["random_state"]),
        init_iterations=int(bank_config["nndsvda_iterations"]),
        device=model.device,
    )
    exemplars = select_exemplars(
        activation,
        samples,
        top_k=int(bank_config["exemplars_per_concept"]),
    )
    exemplars.insert(0, "model_id", MODEL_IDS[backbone])
    exemplar_columns = [
        "model_id",
        "class_idx",
        "concept_idx",
        "exemplar_rank",
        "sample_id",
        "image_id",
        "synset",
        "class_name",
        "rel_path",
        "width",
        "height",
        "patch_id",
        "activation",
    ]
    exemplars = exemplars[exemplar_columns]

    identity = {
        "protocol": "class_specific_full_patch_b9_k8_v1",
        "model_id": MODEL_IDS[backbone],
        "variant": "token_cx",
        "block_number": int(method["evidence_block"]),
        "evidence_layer": int(method["evidence_block"])
        - 1
        - len(model.network.blocks),
        "rank": int(method["rank"]),
        "class_id": int(class_id),
        "sample_count": int(image_count),
        "sample_ids_sha256": value_hash(
            samples["sample_id"].astype(str)
        ),
        "patches_per_image": int(patch_count),
        "token_gradient": True,
        "spatial_weighting": True,
        "spatial_layer": int(method["spatial_block"])
        - 1
        - len(model.network.blocks),
        "spatial_target": str(method["spatial_target"]),
        "nmf_epochs": int(bank_config["nmf_epochs"]),
        "nmf_init_iters": int(bank_config["nndsvda_iterations"]),
        "nmf_random_state": int(bank_config["random_state"]),
        "pretrained_tag": str(model.network.pretrained_cfg["tag"]),
        "bank_type": "class_specific",
    }
    diagnostic_frame = pd.DataFrame(
        [
            {
                "model_id": MODEL_IDS[backbone],
                "class_idx": class_id,
                "identity_sha256": identity_hash(identity),
                **diagnostics,
            }
        ]
    )
    metadata = {
        "schema_version": 1,
        "protocol": "class_specific_full_patch_b9_k8_v1",
        "backbone": backbone,
        "model_id": MODEL_IDS[backbone],
        "model_label": config["models"][backbone]["display_name"],
        "class_id": class_id,
        "rank": int(method["rank"]),
        "block_number": int(method["evidence_block"]),
        "spatial_block": int(method["spatial_block"]),
        "spatial_target": str(method["spatial_target"]),
        "bank_images_per_class": int(image_count),
        "patches_per_image": int(patch_count),
        "exemplars_per_concept": int(
            bank_config["exemplars_per_concept"]
        ),
        "bank_manifest_sha256": manifest_hash,
        "artifact_revision": config["artifacts"]["revision"],
        "diagnostics": diagnostics,
        "identity": identity,
        "identity_sha256": identity_hash(identity),
    }

    temporary = output_root / (
        f".class_{class_id:04d}_{uuid.uuid4().hex}"
    )
    try:
        save_bank_artifacts(
            temporary,
            bases=basis[None],
            class_indices=torch.tensor([class_id]),
            exemplars=exemplars,
            metadata=metadata,
            diagnostics=diagnostic_frame,
        )
        temporary.replace(final_directory)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return "built"


def main():
    args = parse_args()
    config = read_config(args.config)
    images_per_class = int(config["bank"]["images_per_class"])
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not 0 <= args.class_start < args.class_stop <= 1000:
        raise ValueError("class range must be inside [0, 1000]")
    if not args.imagenet_root.exists():
        raise FileNotFoundError(args.imagenet_root)

    output_root = args.output or Path("outputs/banks") / args.backbone
    output_root.mkdir(parents=True, exist_ok=True)
    manifest, manifest_hash = fixed_manifest(config, args.artifact_dir)
    validate_manifest(manifest, images_per_class)
    model = load_model(args.backbone, args.config, device=args.device)

    built = 0
    skipped = 0
    classes = range(args.class_start, args.class_stop)
    for class_id in tqdm(classes, desc="class banks", unit="class"):
        samples = manifest[manifest["class_idx"] == class_id].copy()
        result = build_class(
            model=model,
            config=config,
            backbone=args.backbone,
            class_id=class_id,
            samples=samples,
            manifest_hash=manifest_hash,
            imagenet_root=args.imagenet_root,
            output_root=output_root,
            batch_size=args.batch_size,
        )
        built += result == "built"
        skipped += result == "skipped"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"output={output_root.resolve()}")
    print(f"built={built} skipped={skipped}")
    print(f"bank_manifest_sha256={manifest_hash}")


if __name__ == "__main__":
    main()
