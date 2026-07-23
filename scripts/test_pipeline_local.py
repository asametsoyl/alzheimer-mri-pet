"""End-to-end local pipeline test script.

Runs the complete MRI->PET synthesis pipeline locally on synthetic test data.
Executes all 10 milestones in sequence:
  1. Config loading & validation
  2. Dataset discovery (OASIS adapter) & inspection
  3. Subject-level split & leakage verification
  4. Preprocessing pipeline & caching
  5. PyTorch Dataset & DataLoader
  6. Model & Loss initialization
  7. 1-Epoch training loop
  8. Sliding window inference & 6-metric evaluation
  9. Visualizations (training curves, axial overlays)
 10. Experiment tracking
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import nibabel as nib
import numpy as np
import torch

from config import load_config
from data.adapters.oasis import OASISAdapter
from data.inspector import generate_inspection_report
from data.splitter import split_subjects, verify_no_leakage
from data.dataset import build_dataloader
from engine.colab import report_hardware, set_reproducibility
from evaluation.metrics import compute_all_metrics, aggregate_metrics
from inference.sliding_window import run_inference
from losses.combined import CombinedLoss
from models.factory import build_model
from preprocessing.pipeline import run_preprocessing
from tracking.tracker import ExperimentTracker
from training.trainer import Trainer
from visualization.plots import plot_overlay, plot_training_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("local_test")


def main() -> None:
    logger.info("=== STARTING LOCAL PIPELINE TEST ===")

    # 1. Config & Environment
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    cfg = load_config(cfg_path)

    # Local overrides for fast CPU/GPU test
    cfg.environment.data_root = "data/sample_dataset"
    cfg.environment.cache_root = "outputs/test_cache"
    cfg.environment.output_root = "outputs/test_run"
    cfg.data.dataset = "OASIS"
    cfg.training.max_epochs = 1
    cfg.training.batch_size = 1
    cfg.training.num_workers = 0
    cfg.patch.size = [64, 64, 64]  # Smaller patch for quick CPU/GPU test
    cfg.patch.stride_train = [32, 32, 32]
    cfg.patch.stride_inference = [32, 32, 32]
    cfg.preprocessing.mri.target_shape = [64, 64, 64]

    set_reproducibility(cfg.reproducibility.random_seed, False, False)
    report_hardware()

    # 2. Dataset Discovery & Inspection
    adapter = OASISAdapter()
    data_root = Path(cfg.environment.data_root)
    records = adapter.discover_subjects(data_root)
    logger.info("Discovered %d subjects.", len(records))

    generate_inspection_report(records, "OASIS-Test", sample_size=2)

    # 3. Subject-level Split & Leakage Check
    # With 3 subjects, ratio 0.34 / 0.33 / 0.33 -> 1 train, 1 val, 1 test
    split = split_subjects(records, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, seed=42)
    verify_no_leakage(split)

    # 4. Preprocessing
    all_records = split.train + split.val + split.test
    all_entries = [r.to_dict() for r in all_records]
    preproc_results = run_preprocessing(all_entries, cfg, force=True)
    logger.info("Preprocessing completed for %d subjects.", len(preproc_results))

    # 5. Dataset & DataLoaders
    cache_root = Path(cfg.environment.cache_root)
    train_loader = build_dataloader(split.train, cache_root, "train", cfg)
    val_loader = build_dataloader(split.val, cache_root, "val", cfg)

    batch = next(iter(train_loader))
    logger.info("Batch MRI shape: %s, PET shape: %s", batch["mri"].shape, batch["pet"].shape)

    # 6. Model & Loss Initialisation
    model = build_model(cfg.model)
    loss_fn = CombinedLoss(cfg.loss)
    logger.info("Model built with %d parameters.", model.get_num_parameters())

    # 7. Training Loop (1 Epoch)
    run_dir = Path(cfg.environment.output_root) / "run_local"
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(run_dir, cfg.model_dump())

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_dir=run_dir,
    )
    trainer.train()
    logger.info("1-Epoch Training Completed Successfully!")

    # 8. Inference & Evaluation on Test Set
    test_subject = split.test[0]
    mri_path = cache_root / "preprocessed" / test_subject.subject_id / test_subject.visit_id / "mri.nii.gz"
    pet_path = cache_root / "preprocessed" / test_subject.subject_id / test_subject.visit_id / "pet.nii.gz"

    inf_dir = run_dir / "inference"
    inf_result = run_inference(
        model=model,
        mri_path=mri_path,
        output_dir=inf_dir,
        cfg=cfg,
        subject_id=test_subject.subject_id,
        pet_path=pet_path,
    )

    synth_img = nib.load(inf_result["synth_pet_path"]).get_fdata(dtype=np.float32)
    real_img = nib.load(str(pet_path)).get_fdata(dtype=np.float32)
    mri_img = nib.load(str(mri_path)).get_fdata(dtype=np.float32)
    brain_mask = (mri_img != 0).astype(np.uint8)

    metrics = compute_all_metrics(synth_img, real_img, brain_mask)
    logger.info("Test Evaluation Metrics for %s:", test_subject.subject_id)
    for k, v in metrics.items():
        logger.info("  %-8s: %.4f", k.upper(), v)

    # 9. Visualizations
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    metrics_jsonl = run_dir / "logs" / "metrics.jsonl"
    if metrics_jsonl.exists():
        plot_training_curves(metrics_jsonl, viz_dir)

    overlay_path = viz_dir / f"{test_subject.subject_id}_overlay.png"
    plot_overlay(mri_img, real_img, synth_img, overlay_path, subject_id=test_subject.subject_id)
    logger.info("Saved axial overlay plot to %s", overlay_path)

    tracker.finish()
    logger.info("=== LOCAL PIPELINE TEST COMPLETED SUCCESSFULLY! ALL 10 MILESTONES PASSED ===")


if __name__ == "__main__":
    main()
