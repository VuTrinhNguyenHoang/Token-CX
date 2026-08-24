"""Evaluate Token-CX on the fixed ImageNet validation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import torch
import yaml
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from token_cx import BankRepository, TokenCX, load_model
from token_cx.evaluation import evaluate_saliency, pointing_game
from token_cx.models import read_config


MANIFEST_FILE = "manifests/evaluation_samples.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("deit", "vit"), required=True)
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bank-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/token_cx.yaml"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(".artifacts"),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--metric-batch-size", type=int)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_manifest(config: dict, cache_dir: Path) -> tuple[Path, str]:
    artifact = config["artifacts"]
    common = {
        "repo_id": artifact["repo_id"],
        "repo_type": "dataset",
        "revision": artifact["revision"],
        "local_dir": cache_dir,
    }
    path = Path(
        hf_hub_download(
            filename=MANIFEST_FILE,
            **common,
        )
    )
    artifact_manifest = Path(
        hf_hub_download(
            filename="artifact_manifest.json",
            **common,
        )
    )
    expected = json.loads(
        artifact_manifest.read_text(encoding="utf-8")
    )["evaluation_manifest_sha256"]
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"Evaluation manifest SHA-256 mismatch: {actual}"
        )
    return path, actual


def load_manifest(config: dict, cache_dir: Path) -> tuple[pd.DataFrame, str]:
    path, manifest_hash = download_manifest(config, cache_dir)
    frame = pd.read_csv(
        path,
        dtype={"sample_id": str, "image_id": str},
    )
    counts = frame.groupby("class_idx").size()
    if len(frame) != 5_000 or len(counts) != 1_000 or not counts.eq(5).all():
        raise ValueError(
            "Expected five evaluation images for each of 1,000 "
            "ImageNet classes"
        )
    return frame.reset_index(drop=True), manifest_hash


def image_path(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def bounding_boxes(root: Path, image_id: str) -> list[tuple[int, int, int, int]]:
    annotation = (
        root
        / "ILSVRC"
        / "Annotations"
        / "CLS-LOC"
        / "val"
        / f"{image_id}.xml"
    )
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    boxes = []
    for node in ET.parse(annotation).getroot().findall("object/bndbox"):
        boxes.append(
            tuple(
                int(float(node.findtext(name)))
                for name in ("xmin", "ymin", "xmax", "ymax")
            )
        )
    if not boxes:
        raise ValueError(f"No bounding box in {annotation}")
    return boxes


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_or_validate_config(path: Path, resolved: dict) -> None:
    if path.exists():
        previous = yaml.safe_load(path.read_text())
        if previous != resolved:
            raise ValueError(f"Existing run configuration differs: {path}")
        return
    path.write_text(yaml.safe_dump(resolved, sort_keys=False))


def summarize(
    results: pd.DataFrame,
    output: Path,
    banks: BankRepository,
    config: dict,
    backbone: str,
) -> None:
    metrics = ["deletion_auc", "insertion_auc", "pointing_game"]
    per_class = results.groupby(
        ["candidate", "class_idx"],
        as_index=False,
    )[metrics].mean()
    atomic_csv(per_class, output / "per_class.csv")

    row = {
        "candidate": "token_cx",
        "images": len(results),
        "classes": results["class_idx"].nunique(),
        "classification_accuracy": results["correct"].mean(),
        "query_convergence_rate": results["query_converged"].mean(),
        "backbone": config["models"][backbone]["display_name"],
        "block_number": int(config["method"]["evidence_block"]),
        "k": int(config["method"]["rank"]),
    }
    for metric in metrics:
        row[metric] = per_class[metric].mean()
        row[f"{metric}_ci95"] = 1.96 * per_class[metric].sem()
    row["query_relative_error"] = math.sqrt(
        results["query_reconstruction_squared_error"].sum()
        / results["query_input_squared_norm"].sum()
    )
    if banks.diagnostics is not None:
        row["bank_relative_error"] = math.sqrt(
            banks.diagnostics["reconstruction_squared_error"].sum()
            / banks.diagnostics["input_squared_norm"].sum()
        )
    else:
        row["bank_relative_error"] = float("nan")
    summary = pd.DataFrame([row])
    atomic_csv(summary, output / "summary.csv")
    atomic_csv(summary, output / "table1_row.csv")


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    evaluation = config["evaluation"]
    steps = args.steps or int(evaluation["steps"])
    metric_batch_size = args.metric_batch_size or int(evaluation["batch_size"])
    if steps < 1 or metric_batch_size < 1 or args.save_every < 1:
        raise ValueError(
            "steps, metric-batch-size, and save-every must be positive"
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if not args.imagenet_root.is_dir():
        raise FileNotFoundError(args.imagenet_root)
    output = args.output or Path("outputs/evaluation") / args.backbone
    output.mkdir(parents=True, exist_ok=True)

    manifest, manifest_hash = load_manifest(config, args.artifact_dir)
    if args.limit is not None:
        manifest = manifest.iloc[: args.limit].copy()

    resolved = {
        "backbone": args.backbone,
        "protocol": "class_specific_full_patch_b9_k8_v1",
        "evidence_block": int(config["method"]["evidence_block"]),
        "rank": int(config["method"]["rank"]),
        "manifest": MANIFEST_FILE,
        "manifest_sha256": manifest_hash,
        "bank_source": str(args.bank_dir.resolve())
        if args.bank_dir
        else config["artifacts"]["repo_id"],
        "artifact_revision": None
        if args.bank_dir
        else config["artifacts"]["revision"],
        "steps": steps,
        "metric_batch_size": metric_batch_size,
        "limit": args.limit,
    }
    write_or_validate_config(output / "resolved_config.yaml", resolved)
    manifest.to_csv(output / "evaluation_manifest.csv", index=False)

    model = load_model(args.backbone, args.config, device=args.device)
    if args.bank_dir:
        banks = BankRepository.from_directory(
            args.backbone,
            args.bank_dir,
            args.config,
        )
    else:
        banks = BankRepository.from_config(args.backbone, args.config, args.artifact_dir)
    explainer = TokenCX(model, banks, args.config)

    metric_model = model.network
    if (
        model.device.type == "cuda"
        and model.device.index in (None, 0)
        and torch.cuda.device_count() > 1
    ):
        metric_model = torch.nn.DataParallel(metric_model)

    result_path = output / "per_sample.csv"
    if result_path.exists():
        results = pd.read_csv(result_path, dtype={"sample_id": str, "image_id": str})
    else:
        results = pd.DataFrame()
    required = {
        "sample_id",
        "query_reconstruction_squared_error",
        "query_input_squared_norm",
    }
    if not results.empty and not required <= set(results):
        raise ValueError(f"Incompatible resume file: {result_path}")
    if not results.empty and results["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate samples in {result_path}")
    completed = set(results.get("sample_id", pd.Series(dtype=str)).astype(str))
    declared = set(manifest["sample_id"].astype(str))
    if not completed <= declared:
        raise ValueError(f"Resume rows do not follow {MANIFEST_FILE}")
    pending = manifest[~manifest["sample_id"].astype(str).isin(completed)]

    rows = []
    progress = tqdm(
        pending.itertuples(index=False),
        total=len(pending),
        desc="evaluation",
        unit="image",
    )
    for record in progress:
        path = image_path(args.imagenet_root, record.rel_path)
        explanation = explainer.explain(path)
        query = model.prepare(path)
        scores = evaluate_saliency(
            metric_model,
            query,
            explanation.saliency,
            explanation.target_class,
            steps=steps,
            batch_size=metric_batch_size,
        )
        boxes = bounding_boxes(args.imagenet_root, record.image_id)
        rows.append(
            {
                "candidate": "token_cx",
                "sample_id": str(record.sample_id),
                "image_id": str(record.image_id),
                "class_idx": int(record.class_idx),
                "target": explanation.target_class,
                "target_score": explanation.target_score,
                "correct": int(explanation.target_class == int(record.class_idx)),
                "deletion_auc": scores["deletion_auc"],
                "insertion_auc": scores["insertion_auc"],
                "pointing_game": pointing_game(
                    explanation.saliency,
                    boxes,
                    original_size=(int(record.width), int(record.height)),
                ),
                "query_relative_error": explanation.diagnostics["relative_error"],
                "query_reconstruction_squared_error": explanation.diagnostics[
                    "reconstruction_squared_error"
                ],
                "query_input_squared_norm": explanation.diagnostics[
                    "input_squared_norm"
                ],
                "query_iterations": explanation.diagnostics["iterations"],
                "query_converged": explanation.diagnostics["converged"],
            }
        )
        if len(rows) >= args.save_every:
            results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
            atomic_csv(results, result_path)
            rows.clear()
            progress.set_postfix(saved=len(results))

    if rows:
        results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
        atomic_csv(results, result_path)
    if results.empty:
        raise RuntimeError("No evaluation rows were produced")
    summarize(results, output, banks, config, args.backbone)
    (output / "COMPLETED.json").write_text(
        json.dumps(
            {
                "images": len(results),
                "classes": results["class_idx"].nunique(),
            },
            indent=2,
        )
    )
    print(pd.read_csv(output / "summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
