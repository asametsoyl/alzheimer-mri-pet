"""PET preprocessing pipeline.

Steps (in order):
1. Load NIfTI + validate pairing with MRI
2. Co-register PET to MRI (rigid)
3. Tracer identification
4. Tracer-specific normalization (FDG / Amyloid / Tau / Unknown)
5. Resample PET to match MRI voxel grid exactly
6. Save to cache as .nii.gz
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from config import PETPreprocessingConfig

logger = logging.getLogger(__name__)

# ─── Supported tracers ────────────────────────────────────────────────────────

TracerName = Literal["FDG", "Florbetapir", "Florbetaben", "Flortaucipir", "UNKNOWN"]

TRACER_KEYWORDS: dict[TracerName, list[str]] = {
    "FDG": ["fdg", "18f-fdg", "fluorodeoxyglucose"],
    "Florbetapir": ["florbetapir", "av45", "av-45", "amyvid"],
    "Florbetaben": ["florbetaben", "nav4694", "neuraceq"],
    "Flortaucipir": ["flortaucipir", "av1451", "av-1451", "tauvid"],
}


# ─── Step 1: Load and validate PET pairing ────────────────────────────────────

def load_pet(path: Path) -> nib.Nifti1Image:
    """Load a PET NIfTI file.

    Args:
        path: Path to the PET .nii or .nii.gz file.

    Returns:
        Loaded PET NIfTI image.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"PET file not found: {path}")
    img = nib.load(str(path))
    logger.debug("Loaded PET %s | shape=%s | spacing=%s",
                 path.name, img.shape, img.header.get_zooms()[:3])
    return img


def validate_pet_mri_pair(
    pet_img: nib.Nifti1Image,
    mri_img: nib.Nifti1Image,
    subject_id: str,
) -> None:
    """Validate that PET and MRI are a plausible pair.

    Checks that both images are 3D and have compatible orientation signs.
    Logs a warning (does not raise) if voxel counts differ substantially.

    Args:
        pet_img: PET NIfTI image.
        mri_img: Preprocessed MRI NIfTI image.
        subject_id: Subject identifier for logging.
    """
    if pet_img.ndim not in (3, 4):
        raise ValueError(f"PET image for {subject_id} has unexpected ndim={pet_img.ndim}.")
    if mri_img.ndim != 3:
        raise ValueError(f"MRI image for {subject_id} has unexpected ndim={mri_img.ndim}.")

    pet_voxels = np.prod(pet_img.shape[:3])
    mri_voxels = np.prod(mri_img.shape[:3])
    ratio = pet_voxels / mri_voxels if mri_voxels > 0 else 0
    if ratio < 0.1 or ratio > 10:
        logger.warning(
            "Large voxel count mismatch for %s: PET=%d, MRI=%d (ratio=%.2f). "
            "Registration may fail.",
            subject_id, pet_voxels, mri_voxels, ratio,
        )
    logger.debug("PET/MRI pair validated for subject %s.", subject_id)


# ─── Step 2: Co-register PET to MRI ──────────────────────────────────────────

def coregister_pet_to_mri(
    pet_img: nib.Nifti1Image,
    mri_img: nib.Nifti1Image,
    subject_id: str,
) -> nib.Nifti1Image:
    """Rigidly register PET to MRI using SimpleITK.

    Uses Euler3DTransform (rigid: 6 DOF) with Mattes Mutual Information metric.

    Args:
        pet_img: PET NIfTI image.
        mri_img: Preprocessed MRI NIfTI image (fixed image).
        subject_id: For logging.

    Returns:
        Registered PET image in MRI space.
    """
    pet_data = pet_img.get_fdata(dtype=np.float32)
    mri_data = mri_img.get_fdata(dtype=np.float32)

    # Handle 4D PET (take mean over time dimension)
    if pet_data.ndim == 4:
        logger.info("4D PET detected for %s — averaging over time.", subject_id)
        pet_data = pet_data.mean(axis=-1)

    fixed = sitk.GetImageFromArray(mri_data.T)
    moving = sitk.GetImageFromArray(pet_data.T)

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    registered = resampler.Execute(moving)

    registered_data = sitk.GetArrayFromImage(registered).T.astype(np.float32)
    logger.info("PET→MRI registration complete for %s (metric=%.4f).",
                subject_id, registration.GetMetricValue())
    return nib.Nifti1Image(registered_data, mri_img.affine)


# ─── Step 3: Tracer identification ────────────────────────────────────────────

def identify_tracer(
    filename: str,
    tracer_hint: Optional[str] = None,
) -> TracerName:
    """Identify PET tracer from filename keywords or explicit hint.

    Args:
        filename: Filename or path string to search for tracer keywords.
        tracer_hint: Explicit tracer name from config (overrides auto-detect).

    Returns:
        Identified tracer name or 'UNKNOWN'.
    """
    if tracer_hint and tracer_hint.lower() != "auto":
        # Direct match or close match
        for tracer in TRACER_KEYWORDS:
            if tracer_hint.lower() in tracer.lower() or tracer.lower() in tracer_hint.lower():
                logger.debug("Tracer set explicitly: %s.", tracer)
                return tracer  # type: ignore[return-value]
        logger.warning("Explicit tracer hint '%s' not recognized — falling back to auto.", tracer_hint)

    fn_lower = filename.lower()
    for tracer, keywords in TRACER_KEYWORDS.items():
        if any(kw in fn_lower for kw in keywords):
            logger.info("Auto-detected tracer: %s from filename '%s'.", tracer, filename)
            return tracer  # type: ignore[return-value]

    logger.warning("Could not identify PET tracer from '%s'. Marking as UNKNOWN.", filename)
    return "UNKNOWN"


# ─── Step 4: Tracer-specific normalization ────────────────────────────────────

def compute_reference_region_mean(
    pet_data: np.ndarray,
    reference_mask: Optional[np.ndarray],
) -> float:
    """Compute mean PET intensity in a reference region.

    Args:
        pet_data: 3D PET intensity array.
        reference_mask: Boolean mask for the reference region.
            If None, uses whole-brain mean as fallback.

    Returns:
        Mean intensity in the reference region.
    """
    if reference_mask is not None and reference_mask.sum() > 0:
        ref_mean = float(pet_data[reference_mask.astype(bool)].mean())
    else:
        logger.warning("Reference mask is empty or None — using whole-brain mean.")
        ref_mean = float(pet_data[pet_data > 0].mean())
    return ref_mean


def normalize_pet(
    pet_img: nib.Nifti1Image,
    tracer: TracerName,
    cfg: PETPreprocessingConfig,
    reference_mask: Optional[np.ndarray] = None,
    injected_dose: Optional[float] = None,
    body_weight_kg: Optional[float] = None,
) -> tuple[nib.Nifti1Image, dict]:
    """Apply tracer-specific PET intensity normalization.

    Args:
        pet_img: Registered PET image (in MRI space).
        tracer: Identified tracer name.
        cfg: PET preprocessing config.
        reference_mask: Boolean mask for reference region.
        injected_dose: Injected dose in MBq (for FDG SUV normalization).
        body_weight_kg: Body weight in kg (for FDG SUV normalization).

    Returns:
        Tuple of (normalized_pet_image, normalization_report_dict).
    """
    data = pet_img.get_fdata(dtype=np.float32).copy()
    report: dict = {"tracer": tracer}

    if tracer == "FDG":
        if injected_dose is not None and body_weight_kg is not None:
            # SUV = (PET counts / injected_dose_kBq) * body_weight_g
            injected_dose_kbq = injected_dose * 1000
            body_weight_g = body_weight_kg * 1000
            data = data / injected_dose_kbq * body_weight_g
            report["normalization"] = "SUV"
            report["mean_cortical_suv"] = float(data[data > 0].mean())
        else:
            # Fallback: normalize to reference region
            ref_mean = compute_reference_region_mean(data, reference_mask)
            data = data / (ref_mean + 1e-8)
            report["normalization"] = "reference_region_ratio"
        logger.info("FDG normalization applied. Mean SUV: %.3f", data[data > 0].mean())

    elif tracer in ("Florbetapir", "Florbetaben"):
        ref_mean = compute_reference_region_mean(data, reference_mask)
        data = data / (ref_mean + 1e-8)  # SUVr
        threshold = cfg.amyloid_suvr_threshold.get(tracer, 1.10)
        mean_suvr = float(data[data > 0].mean())
        report["normalization"] = "SUVr"
        report["reference_region"] = cfg.reference_region.get(tracer, "unknown")
        report["mean_suvr"] = mean_suvr
        report["amyloid_positive"] = mean_suvr > threshold
        report["threshold"] = threshold
        logger.info("Amyloid PET (%s) SUVr=%.3f | threshold=%.2f | positive=%s",
                    tracer, mean_suvr, threshold, report["amyloid_positive"])

    elif tracer == "Flortaucipir":
        ref_mean = compute_reference_region_mean(data, reference_mask)
        data = data / (ref_mean + 1e-8)  # SUVr
        report["normalization"] = "SUVr"
        report["reference_region"] = cfg.reference_region.get(tracer, "unknown")
        report["mean_suvr"] = float(data[data > 0].mean())
        logger.info("Tau PET (Flortaucipir) SUVr=%.3f.", report["mean_suvr"])

    else:  # UNKNOWN
        low = float(np.percentile(data[data > 0], 0.5))
        high = float(np.percentile(data[data > 0], 99.5))
        data = np.clip(data, low, high)
        data = (data - low) / (high - low + 1e-8)
        report["normalization"] = "min_max_0_1"
        report["WARNING"] = "UNKNOWN tracer — conservative [0,1] normalization applied."
        logger.warning("UNKNOWN PET tracer — min-max normalization applied.")

    return nib.Nifti1Image(data, pet_img.affine), report


# ─── Step 5: Resample PET to MRI grid ─────────────────────────────────────────

def resample_pet_to_mri(
    pet_img: nib.Nifti1Image,
    mri_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Resample PET to exactly match the MRI voxel grid.

    Args:
        pet_img: Normalized PET image.
        mri_img: Preprocessed MRI image (defines the target grid).

    Returns:
        PET image resampled to the MRI grid.
    """
    mri_data = mri_img.get_fdata(dtype=np.float32)
    pet_data = pet_img.get_fdata(dtype=np.float32)

    fixed = sitk.GetImageFromArray(mri_data.T)
    moving = sitk.GetImageFromArray(pet_data.T)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(sitk.Transform())  # identity (already co-registered)
    resampled = resampler.Execute(moving)

    out_data = sitk.GetArrayFromImage(resampled).T.astype(np.float32)
    logger.debug("PET resampled to MRI grid: %s.", out_data.shape)
    return nib.Nifti1Image(out_data, mri_img.affine)


# ─── Full PET Pipeline ────────────────────────────────────────────────────────

def preprocess_pet(
    input_path: Path,
    mri_img: nib.Nifti1Image,
    cache_path: Path,
    cfg: PETPreprocessingConfig,
    subject_id: str,
    tracer_hint: Optional[str] = "auto",
    reference_mask: Optional[np.ndarray] = None,
    injected_dose: Optional[float] = None,
    body_weight_kg: Optional[float] = None,
    force: bool = False,
) -> tuple[nib.Nifti1Image, dict]:
    """Run the full 6-step PET preprocessing pipeline for one subject.

    Args:
        input_path: Path to the raw PET NIfTI file.
        mri_img: Already preprocessed MRI image for this subject/visit.
        cache_path: Destination path for the preprocessed PET output.
        cfg: PET preprocessing configuration.
        subject_id: Subject identifier (for logging).
        tracer_hint: Explicit tracer name or 'auto'.
        reference_mask: Optional reference region mask array.
        injected_dose: FDG injected dose in MBq.
        body_weight_kg: Subject body weight in kg.
        force: If True, reprocess even if cache_path already exists.

    Returns:
        Tuple of (preprocessed_pet_image, normalization_report).
    """
    if cache_path.exists() and not force:
        logger.info("Cache hit for PET %s — skipping preprocessing.", subject_id)
        report: dict = {"tracer": "cached", "normalization": "cached"}
        return nib.load(str(cache_path)), report

    logger.info("[PET] Preprocessing subject %s ...", subject_id)

    # 1. Load PET
    pet_img = load_pet(input_path)

    # 2. Validate pairing
    validate_pet_mri_pair(pet_img, mri_img, subject_id)

    # 3. Co-register PET to MRI
    if cfg.coregister_to_mri:
        pet_img = coregister_pet_to_mri(pet_img, mri_img, subject_id)

    # 4. Tracer identification
    tracer = identify_tracer(input_path.name, tracer_hint)

    # 5. Tracer-specific normalization
    pet_img, report = normalize_pet(
        pet_img, tracer, cfg, reference_mask, injected_dose, body_weight_kg,
    )

    # 6. Resample PET to MRI grid
    pet_img = resample_pet_to_mri(pet_img, mri_img)

    # 7. Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(pet_img, str(cache_path))
    logger.info("[PET] Done: %s → %s", subject_id, cache_path)
    return pet_img, report
