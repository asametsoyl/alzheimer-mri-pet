# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Initial project scaffold
- Config system (pydantic-based)
- MRI and PET preprocessing pipelines
- Multi-dataset adapters (ADNI, AIBL, OASIS-3, Custom)
- Subject-level data splitter with leakage verification
- 3D Residual Attention U-Net model
- Combined loss (L1 + 3D SSIM)
- Trainer with AMP + gradient accumulation + checkpoint resume
- Sliding window inference + TTA
- Full evaluation suite (SSIM, PSNR, MAE, MSE, NMSE, PCC + clinical)
- Visualization: training curves, overlays, Bland-Altman
- Experiment tracker with optional W&B integration
- Google Colab notebook (end-to-end)
- Complete documentation
