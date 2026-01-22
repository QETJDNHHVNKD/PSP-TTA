# PSP-TTA
 PSP-TTA

Anonymous code release for the ICML submission **"Polar Shape Priors for Stable Test-Time Adaptation under Cross-Modal Shifts"**.

This repository provides the implementation of **PSP-TTA**, a shape-centric test-time adaptation framework for cross-modality medical image segmentation. The method performs per-sample adaptation by optimizing only shape latents while freezing network weights, with differentiable structural consistency and a gated multi-source shape prior.

## Environment
- Python >= 3.9
- PyTorch >= 1.10
- CUDA (optional, recommended)

```bash


Data

This code supports evaluation on ISIC and PH2.
Please download the datasets from their official sources and organize them as:
data/
  ISIC/...
  PH2/...

## Notes

The repository is anonymized for double-blind review.

Pretrained checkpoints are not included. Please place checkpoints under checkpoints/ (or any path) and specify via --ckpt.

