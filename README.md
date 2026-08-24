# Token-CX

Official implementation of **Token-CX: Token Concept-Based Explanations for Vision Transformers**.

Token-CX explains Vision Transformer predictions using class-specific concept
banks learned from patch-token representations. Given an input image, the method
extracts class-relevant token evidence, performs fixed-basis concept inference,
and produces concept masks, representative training exemplars, and an aggregated
saliency map.

## Overview

Token-CX consists of four main stages:

1. Extract current patch-token representations from an intermediate transformer block.
2. Combine the tokens with the target token gradient and gradient-guided spatial weighting.
3. Infer query-specific concept activations over a fixed class-specific NMF basis.
4. Generate concept masks and aggregate them into the final explanation.

The main paper configuration uses transformer **Block 9** and concept rank
\(K=8\).

## Explanation outputs

For each query image, Token-CX returns:

- the model-predicted target class;
- query-specific concept activations;
- spatial masks associated with the activated concepts;
- representative training exemplars for each concept;
- an activation-weighted Token-CX saliency map.

The concept basis is learned offline and remains fixed during query explanation.
Only the query-specific non-negative activations are inferred at explanation
time.

## Supported models

| Backbone | Model checkpoint |
|---|---|
| ViT-B/16 | `vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| DeiT-B/16-Distilled | `deit_base_distilled_patch16_224.fb_in1k` |

The pretrained models are loaded through
[`timm`](https://github.com/huggingface/pytorch-image-models).

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/VuTrinhNguyenHoang/Token-CX.git
cd Token-CX
pip install -e .
```

To run the tests:

```bash
pip install -e ".[test]"
pytest
```

To access the artifact manifests through Hugging Face Datasets:

```bash
pip install -e ".[data]"
```

## Precomputed artifacts

Precomputed concept banks, exemplar metadata, evaluation manifests, and bank
diagnostics are hosted on Hugging Face:

**[trinhvu21/token-cx-artifacts](https://huggingface.co/datasets/trinhvu21/token-cx-artifacts)**

The configuration in this repository pins the artifact revision used for the
reported experiments:

```text
0a0a878aabd35664cf0885055cccffe8d35413d0
```

Artifacts are provided for:

- ViT-B/16;
- DeiT-B/16-Distilled;
- 1,000 ImageNet classes;
- concept rank \(K=8\).

The repository does not redistribute ImageNet images. Exemplar files contain
identifiers, patch locations, activation values, and crop metadata. Access to
the corresponding ImageNet images is required to reproduce exemplar
visualizations.

## Quick start

```python
from token_cx import BankRepository, TokenCX, load_model

model = load_model(
    backbone="deit",
    config_path="configs/token_cx.yaml",
)

banks = BankRepository.from_config(
    backbone="deit",
    config_path="configs/token_cx.yaml",
)

explainer = TokenCX(
    model=model,
    banks=banks,
    config_path="configs/token_cx.yaml",
)

explanation = explainer.explain("image.jpg")
```

The returned `explanation` object provides:

```python
explanation.target_class
explanation.target_score
explanation.activations
explanation.concept_masks
explanation.saliency
explanation.exemplars
```

The activation-weighted saliency map can be visualized using:

```python
from token_cx.visualization import plot_saliency

plot_saliency(
    image="image.jpg",
    explanation=explanation,
)
```

The paper-style query-to-bank figure additionally requires access to the
ImageNet training images referenced by the exemplar metadata:

```python
from token_cx.visualization import plot_explanation

plot_explanation(
    image="image.jpg",
    explanation=explanation,
    imagenet_root="/path/to/imagenet",
)
```

Concept identifiers are zero-based and match the `concept_idx` values stored
in the artifact repository. The paper demo notebook reproduces the saved
qualitative ordering for Figures 5 and 6 without changing the
activation-weighted Token-CX saliency. It requires the ImageNet Object
Localization Challenge data and renders both figures directly.

## Reproducing the experiments

Both scripts use the immutable manifests and artifact revision declared in
`configs/token_cx.yaml`. ImageNet images are not downloaded by the scripts;
`--imagenet-root` must point to the ImageNet Object Localization Challenge
directory containing `ILSVRC/Data/CLS-LOC` and `ILSVRC/Annotations/CLS-LOC`.

Rebuild the class-specific DeiT concept banks:

```bash
python scripts/build_banks.py \
  --backbone deit \
  --imagenet-root /path/to/imagenet \
  --output outputs/banks/deit
```

Bank construction is resumable at class granularity. Independent jobs can use
`--class-start` and `--class-stop`; their `class_####` directories can then be
placed under the same output root. ViT uses the identical command with
`--backbone vit`.

Evaluate the public precomputed banks on the fixed 5,000-image benchmark:

```bash
python scripts/evaluate.py \
  --backbone deit \
  --imagenet-root /path/to/imagenet \
  --output outputs/evaluation/deit
```

Pass `--bank-dir outputs/banks/deit` to evaluate locally rebuilt banks. For a
quick execution check, add `--limit 10`; limited runs are smoke tests and must
not be reported as paper results. The evaluation writes per-image, per-class,
and class-balanced summary CSV files and can resume from `per_sample.csv`.

## Main results

| Backbone | Deletion AUC ↓ | Insertion AUC ↑ | Pointing Game ↑ |
|---|---:|---:|---:|
| ViT-B/16 | 0.169 | 0.568 | 90.82% |
| DeiT-B/16-Distilled | 0.215 | 0.758 | 90.92% |

Lower Deletion AUC and higher Insertion AUC indicate better faithfulness.
Higher Pointing Game accuracy indicates better spatial localization.

## Repository structure

```text
Token-CX/
├── configs/
│   └── token_cx.yaml
├── src/
│   └── token_cx/
│       ├── models.py
│       ├── method.py
│       ├── banks.py
│       ├── evaluation.py
│       └── visualization.py
├── notebooks/
│   └── demo.ipynb
├── scripts/
│   ├── build_banks.py
│   └── evaluate.py
└── tests/
```

The reusable implementation is contained in `src/token_cx`; the notebook is a
minimal reproduction of the two paper exemplar figures, while the scripts
rebuild the bank artifacts and quantitative results.

## Configuration

The canonical configuration is located at:

```text
configs/token_cx.yaml
```

It contains:

- pretrained model identifiers;
- the pinned Hugging Face artifact revision;
- the selected transformer block;
- concept rank and fixed-basis inference settings;
- bank-construction settings;
- evaluation settings.

The public configuration corresponds to the final method reported in the paper.

## Reproducibility

Token-CX separates offline concept discovery from query-time explanation:

1. Class-specific concept banks are constructed from ImageNet training images.
2. The learned NMF bases are stored and kept fixed.
3. For a query image, Token-CX selects the bank associated with the model-predicted class.
4. Only query-specific concept activations are inferred.
5. Concept masks and the final saliency map are generated from these activations.

The distributed manifests preserve sample identifiers and artifact revisions so
that bank construction and evaluation can be reproduced consistently.

## Citation

If you use Token-CX in your research, please cite the associated paper and this
repository. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

The final BibTeX entry will be added after publication.

## License

The source code is released under the [MIT License](LICENSE).

Precomputed artifacts and ImageNet-derived metadata are distributed separately
through the Hugging Face artifact repository. ImageNet images and pretrained
model weights remain subject to their respective licenses and terms of use.
