# Project Progress

| Milestone | Status | Notes |
|---|---|---|
| M1 — Environment & Scaffold | ✅ DONE | Config, README, requirements, folder structure |
| M2 — Preprocessing Pipeline | ✅ DONE | MRI (9 steps) + PET (6 steps) + orchestrator |
| M3 — Data Module | ✅ DONE | ADNI/AIBL/OASIS/Custom adapters, inspector, splitter, dataset, dataloader |
| M4 — Models | ✅ DONE | BaseModel, ResAttnUNet3D, model factory |
| M5 — Loss Functions | ✅ DONE | MaskedL1, 3D SSIM, GradientDiff, CombinedLoss |
| M6 — Training Loop | ✅ DONE | Trainer (AMP, accum, checkpoint, NaN, OOM) + Colab engine |
| M7 — Evaluation | ✅ DONE | SSIM, PSNR, MAE, MSE, NMSE, PCC + statistical tests |
| M8 — Inference | ✅ DONE | Sliding window (Gaussian) + TTA + difference map |
| M9 — Visualization | ✅ DONE | Training curves, axial overlays, Bland-Altman |
| M10 — Tracking & Docs | ✅ DONE | ExperimentTracker, ADR-001, architecture diagram |
| Tests | 🔄 IN PROGRESS | Unit tests for config, splitter, metrics, model |
| Colab Notebook | ✅ DONE | 12-section end-to-end notebook |

Last updated: 2024-01-15
