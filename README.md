# MRI → PET Synthesis for Alzheimer's Disease

A **publication-quality, production-ready** pipeline for synthesizing PET images from
structural MRI using deep learning — optimized for Google Colab (T4/A100).

---

## Features

- Multi-tracer support: **FDG**, **Amyloid PET** (Florbetapir/Florbetaben), **Tau PET** (Flortaucipir)
- Multi-dataset: **ADNI**, **AIBL**, **OASIS-3**, custom
- Subject-level splitting with **zero-leakage guarantee**
- Tracer-specific SUVr normalization
- Model registry: **3D Residual Attention U-Net**, Attention U-Net, UNETR, Swin UNETR, MedNeXt
- AMP + gradient accumulation + automatic checkpoint resume
- Full evaluation: SSIM, PSNR, MAE, MSE, NMSE, PCC + clinical SUVr metrics
- Sliding window inference with Gaussian weighting + TTA
- Optional Weights & Biases integration

---

## Quick Start (Google Colab)

1. Open `notebooks/MRI_PET_Colab.ipynb` in Google Colab
2. Mount your Google Drive
3. Place your dataset under `/content/drive/MyDrive/Google Colabs/data/`
4. Run all cells

---

## Local Installation

```bash
git clone <repo>
cd alzheimer-mri-pet
pip install -e ".[dev]"
```

---

## Training

```bash
python -m alzheimer_mri_pet.training.trainer \
  --config config/default.yaml \
  --experiment config/experiments/my_run.yaml
```

---

## Inference

```bash
python -m alzheimer_mri_pet.inference.runner \
  --checkpoint outputs/<run_id>/checkpoints/best_model.pt \
  --input path/to/mri.nii.gz \
  --output path/to/output/
```

---

## Citation

```bibtex
@software{alzheimer_mri_pet_2024,
  title  = {MRI to PET Synthesis for Alzheimer's Disease},
  year   = {2024},
}
```

---

## License

MIT — see `LICENSE`. Dataset licenses are documented in `DATA_LICENSE.md`.
