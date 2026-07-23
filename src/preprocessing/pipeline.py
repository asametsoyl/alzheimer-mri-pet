"""Preprocessing orchestrator.

Runs MRI + PET preprocessing for all subjects.
- Handles per-subject errors gracefully (logs + continues).
- Reports all failed subjects at the end.
- Checks cache before processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib

from config import PipelineConfig
from preprocessing.mri import preprocess_mri
from preprocessing.pet import preprocess_pet

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result for a single subject/visit preprocessing run."""

    subject_id: str
    visit_id: str
    mri_cache_path: Optional[Path] = None
    pet_cache_path: Optional[Path] = None
    pet_report: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


def run_preprocessing(
    subjects: list[dict],
    cfg: PipelineConfig,
    force: bool = False,
) -> list[PreprocessingResult]:
    """Run preprocessing for all subjects.

    Each entry in `subjects` is a dict with keys:
        subject_id, visit_id, mri_path, pet_path,
        and optionally: injected_dose, body_weight_kg

    Args:
        subjects: List of subject/visit metadata dicts.
        cfg: Full pipeline configuration.
        force: If True, reprocess even if cached files exist.

    Returns:
        List of PreprocessingResult (one per subject/visit).
    """
    cache_root = Path(cfg.environment.cache_root) / "preprocessed"
    tmp_dir = Path(cfg.environment.cache_root) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: list[PreprocessingResult] = []
    failed: list[str] = []

    for entry in subjects:
        subject_id = entry["subject_id"]
        visit_id = entry.get("visit_id", "ses-baseline")
        mri_path = Path(entry["mri_path"])
        pet_path = Path(entry["pet_path"])

        mri_cache = cache_root / subject_id / visit_id / "mri.nii.gz"
        pet_cache = cache_root / subject_id / visit_id / "pet.nii.gz"
        result = PreprocessingResult(subject_id=subject_id, visit_id=visit_id)

        try:
            # ── MRI ──
            mri_img = preprocess_mri(
                input_path=mri_path,
                cache_path=mri_cache,
                cfg=cfg.preprocessing.mri,
                subject_id=subject_id,
                tmp_dir=tmp_dir,
                force=force,
            )
            result.mri_cache_path = mri_cache

            # ── PET ──
            pet_img, pet_report = preprocess_pet(
                input_path=pet_path,
                mri_img=mri_img,
                cache_path=pet_cache,
                cfg=cfg.preprocessing.pet,
                subject_id=subject_id,
                tracer_hint=cfg.data.pet_tracer,
                injected_dose=entry.get("injected_dose"),
                body_weight_kg=entry.get("body_weight_kg"),
                force=force,
            )
            result.pet_cache_path = pet_cache
            result.pet_report = pet_report

        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Preprocessing FAILED for %s/%s — %s",
                subject_id, visit_id, error_msg,
                exc_info=True,
            )
            result.success = False
            result.error = error_msg
            failed.append(f"{subject_id}/{visit_id}: {error_msg}")

        results.append(result)

    # Summary
    n_total = len(results)
    n_ok = sum(r.success for r in results)
    logger.info(
        "Preprocessing complete: %d/%d succeeded, %d failed.",
        n_ok, n_total, len(failed),
    )
    if failed:
        logger.error("Failed subjects:\n%s", "\n".join(f"  - {s}" for s in failed))

    return results
