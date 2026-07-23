"""MRI preprocessing pipeline.

Steps (in order):
1. Load NIfTI + validate affine
2. Reorient to RAS+ canonical
3. Resample to isotropic 1 mm³
4. Skull stripping (HD-BET or SynthStrip)
5. N4ITK bias field correction
6. MNI152 registration (optional)
7. Intensity normalization (z-score / percentile / whitestripe)
8. Crop or pad to target shape
9. Save to cache as .nii.gz
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from config import MRIPreprocessingConfig

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SINGULAR_AFFINE_THRESHOLD = 1e-6


# ─── Step 1: Load & validate affine ──────────────────────────────────────────

def load_and_validate_nifti(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI file and validate that its affine is non-singular.

    Args:
        path: Path to the .nii or .nii.gz file.

    Returns:
        Loaded NIfTI image.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the affine determinant is near-zero (singular).
    """
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(str(path))
    det = np.linalg.det(img.affine[:3, :3])
    if abs(det) < SINGULAR_AFFINE_THRESHOLD:
        raise ValueError(
            f"Near-singular affine (det={det:.2e}) in {path}. "
            "Check image geometry."
        )
    logger.debug("Loaded %s | shape=%s | spacing=%s | det=%.4f",
                 path.name, img.shape, img.header.get_zooms()[:3], det)
    return img


# ─── Step 2: Reorient to RAS+ ─────────────────────────────────────────────────

def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient image to RAS+ canonical orientation.

    Args:
        img: Input NIfTI image.

    Returns:
        Reoriented NIfTI image in RAS+ orientation.
    """
    canonical = nib.as_closest_canonical(img)
    logger.debug("Reoriented to RAS+ (was %s)", nib.aff2axcodes(img.affine))
    return canonical


# ─── Step 3: Resample to isotropic spacing ────────────────────────────────────

def resample_to_spacing(
    img: nib.Nifti1Image,
    target_spacing: list[float],
    interpolator: int = sitk.sitkLinear,
) -> nib.Nifti1Image:
    """Resample a NIfTI image to a target isotropic voxel spacing.

    Args:
        img: Input NIfTI image (in RAS+ orientation).
        target_spacing: Desired voxel spacing in mm, e.g. [1.0, 1.0, 1.0].
        interpolator: SimpleITK interpolator constant.

    Returns:
        Resampled NIfTI image.
    """
    data = img.get_fdata(dtype=np.float32)
    sitk_img = sitk.GetImageFromArray(data.T)  # NIfTI is Fortran-order
    sitk_img.SetSpacing([float(s) for s in img.header.get_zooms()[:3]])
    sitk_img.SetOrigin([0.0, 0.0, 0.0])

    original_size = np.array(sitk_img.GetSize())
    original_spacing = np.array(sitk_img.GetSpacing())
    new_spacing = np.array(target_spacing, dtype=np.float64)
    new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size)
    resampler.SetInterpolator(interpolator)
    resampled = resampler.Execute(sitk_img)

    new_data = sitk.GetArrayFromImage(resampled).T.astype(np.float32)
    new_zooms = target_spacing
    new_affine = np.diag([*new_zooms, 1.0])
    new_affine[:3, 3] = img.affine[:3, 3]  # preserve origin

    logger.debug("Resampled %s → %s | spacing %s → %s",
                 data.shape, new_data.shape, original_spacing.tolist(), target_spacing)
    return nib.Nifti1Image(new_data, new_affine)


# ─── Step 4: Skull stripping ──────────────────────────────────────────────────

def skull_strip(
    img: nib.Nifti1Image,
    tool: str,
    subject_id: str,
    output_dir: Path,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Apply skull stripping to remove non-brain tissue.

    Attempts the preferred tool; falls back to the next if unavailable.

    Args:
        img: Resampled MRI image.
        tool: One of 'hd-bet', 'synthstrip', 'none'.
        subject_id: Used for logging and temp file naming.
        output_dir: Directory to write intermediate files if needed.

    Returns:
        Tuple of (stripped_image, brain_mask).
    """
    if tool == "none":
        logger.warning("Skull stripping disabled for %s.", subject_id)
        mask_data = (img.get_fdata() > 0).astype(np.uint8)
        mask = nib.Nifti1Image(mask_data, img.affine)
        return img, mask

    # Try HD-BET
    if tool == "hd-bet":
        try:
            return _skull_strip_hdbet(img, subject_id, output_dir)
        except Exception as e:
            logger.warning("HD-BET failed for %s (%s). Falling back to SynthStrip.", subject_id, e)

    # Try SynthStrip
    try:
        return _skull_strip_synthstrip(img, subject_id, output_dir)
    except Exception as e:
        logger.error("SynthStrip also failed for %s (%s). Returning unstripped image.", subject_id, e)
        mask_data = (img.get_fdata() > 0).astype(np.uint8)
        return img, nib.Nifti1Image(mask_data, img.affine)


def _skull_strip_hdbet(
    img: nib.Nifti1Image,
    subject_id: str,
    output_dir: Path,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Skull strip using HD-BET (requires hd-bet package).

    Args:
        img: Input MRI image.
        subject_id: Subject identifier for file naming.
        output_dir: Directory to write HD-BET outputs.

    Returns:
        Tuple of (stripped_image, brain_mask).
    """
    import subprocess  # noqa: PLC0415

    tmp_in = output_dir / f"{subject_id}_tmp_input.nii.gz"
    tmp_out = output_dir / f"{subject_id}_tmp_hdbet"
    nib.save(img, str(tmp_in))
    subprocess.run(
        ["hd-bet", "-i", str(tmp_in), "-o", str(tmp_out), "-mode", "fast", "-s", "1"],
        check=True, capture_output=True,
    )
    stripped = nib.load(str(tmp_out) + ".nii.gz")
    mask = nib.load(str(tmp_out) + "_mask.nii.gz")
    tmp_in.unlink(missing_ok=True)
    logger.info("HD-BET skull stripping completed for %s.", subject_id)
    return stripped, mask


def _skull_strip_synthstrip(
    img: nib.Nifti1Image,
    subject_id: str,
    output_dir: Path,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Skull strip using SynthStrip (FreeSurfer).

    Args:
        img: Input MRI image.
        subject_id: Subject identifier for file naming.
        output_dir: Directory to write SynthStrip outputs.

    Returns:
        Tuple of (stripped_image, brain_mask).
    """
    import subprocess  # noqa: PLC0415

    tmp_in = output_dir / f"{subject_id}_tmp_input.nii.gz"
    tmp_out = output_dir / f"{subject_id}_tmp_stripped.nii.gz"
    tmp_mask = output_dir / f"{subject_id}_tmp_mask.nii.gz"
    nib.save(img, str(tmp_in))
    subprocess.run(
        ["mri_synthstrip", "-i", str(tmp_in), "-o", str(tmp_out), "-m", str(tmp_mask)],
        check=True, capture_output=True,
    )
    stripped = nib.load(str(tmp_out))
    mask = nib.load(str(tmp_mask))
    tmp_in.unlink(missing_ok=True)
    logger.info("SynthStrip skull stripping completed for %s.", subject_id)
    return stripped, mask


# ─── Step 5: Bias field correction ────────────────────────────────────────────

def correct_bias_field(img: nib.Nifti1Image, mask: Optional[nib.Nifti1Image] = None) -> nib.Nifti1Image:
    """Apply N4ITK bias field correction via SimpleITK.

    Args:
        img: Input MRI image (skull-stripped preferred).
        mask: Optional brain mask to restrict correction.

    Returns:
        Bias-corrected NIfTI image.
    """
    data = img.get_fdata(dtype=np.float32)
    sitk_img = sitk.GetImageFromArray(data.T)
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

    sitk_mask = None
    if mask is not None:
        mask_data = mask.get_fdata().astype(np.uint8)
        sitk_mask = sitk.GetImageFromArray(mask_data.T)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_img, sitk_mask) if sitk_mask else corrector.Execute(sitk_img)

    corrected_data = sitk.GetArrayFromImage(corrected).T.astype(np.float32)
    logger.debug("N4ITK bias correction applied.")
    return nib.Nifti1Image(corrected_data, img.affine)


# ─── Step 7: Intensity normalization ──────────────────────────────────────────

def normalize_intensity(
    img: nib.Nifti1Image,
    method: str,
    mask: Optional[nib.Nifti1Image] = None,
    percentile_clip: list[float] | None = None,
) -> nib.Nifti1Image:
    """Normalize MRI intensity values.

    Args:
        img: Input MRI image.
        method: One of 'z-score', 'percentile', 'whitestripe'.
        mask: Optional brain mask; normalization statistics computed inside mask.
        percentile_clip: [low, high] percentiles; used only when method='percentile'.

    Returns:
        Intensity-normalized NIfTI image.
    """
    data = img.get_fdata(dtype=np.float32).copy()

    if mask is not None:
        mask_data = mask.get_fdata().astype(bool)
        brain_vals = data[mask_data]
    else:
        brain_vals = data[data > 0]

    if method == "z-score":
        mean = float(brain_vals.mean())
        std = float(brain_vals.std())
        if std < 1e-8:
            logger.warning("Near-zero std (%.2e) during z-score normalization.", std)
            std = 1.0
        data = (data - mean) / std

    elif method == "percentile":
        clips = percentile_clip or [0.5, 99.5]
        low = float(np.percentile(brain_vals, clips[0]))
        high = float(np.percentile(brain_vals, clips[1]))
        data = np.clip(data, low, high)
        data = (data - low) / (high - low + 1e-8)

    elif method == "whitestripe":
        # WhiteStripe: normalize to white matter mode (simplified version)
        # Full implementation requires the whitestripe package or histogram peak detection
        hist, bins = np.histogram(brain_vals, bins=1000)
        wm_peak = float(bins[np.argmax(hist)])
        std = float(brain_vals.std())
        data = (data - wm_peak) / (std + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    logger.debug("Intensity normalization '%s' applied. Range: [%.3f, %.3f]",
                 method, float(data.min()), float(data.max()))
    return nib.Nifti1Image(data, img.affine)


# ─── Step 8: Crop / pad to target shape ───────────────────────────────────────

def crop_or_pad(img: nib.Nifti1Image, target_shape: list[int]) -> nib.Nifti1Image:
    """Center-crop or symmetrically pad image to a fixed target shape.

    Args:
        img: Input NIfTI image.
        target_shape: Desired output shape [D, H, W].

    Returns:
        Image with exact target_shape.
    """
    data = img.get_fdata(dtype=np.float32)
    current_shape = np.array(data.shape[:3])
    target = np.array(target_shape)

    # Pad if any dimension is smaller than target
    pad_before = np.maximum(0, (target - current_shape) // 2)
    pad_after = np.maximum(0, target - current_shape - pad_before)
    pad_width = [(int(b), int(a)) for b, a in zip(pad_before, pad_after)]
    if len(data.shape) == 4:
        pad_width.append((0, 0))
    data = np.pad(data, pad_width, mode="constant", constant_values=0)

    # Crop if any dimension is larger than target
    current_shape = np.array(data.shape[:3])
    start = (current_shape - target) // 2
    slices = tuple(slice(int(s), int(s + t)) for s, t in zip(start, target))
    data = data[slices]

    logger.debug("Crop/pad → target shape %s.", target_shape)
    return nib.Nifti1Image(data, img.affine)


# ─── Step 9: Save to cache ────────────────────────────────────────────────────

def save_to_cache(img: nib.Nifti1Image, cache_path: Path) -> None:
    """Save a NIfTI image to the local cache.

    Args:
        img: NIfTI image to save.
        cache_path: Full output path (.nii.gz).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(cache_path))
    logger.debug("Saved to cache: %s", cache_path)


# ─── Full MRI Pipeline ────────────────────────────────────────────────────────

def preprocess_mri(
    input_path: Path,
    cache_path: Path,
    cfg: MRIPreprocessingConfig,
    subject_id: str,
    tmp_dir: Path,
    force: bool = False,
) -> nib.Nifti1Image:
    """Run the full 9-step MRI preprocessing pipeline for one subject.

    Args:
        input_path: Path to the raw MRI NIfTI file.
        cache_path: Destination path for the preprocessed output.
        cfg: MRI preprocessing configuration.
        subject_id: Subject identifier (for logging).
        tmp_dir: Temporary directory for skull stripping intermediates.
        force: If True, reprocess even if cache_path already exists.

    Returns:
        Preprocessed MRI as a NIfTI image.
    """
    if cache_path.exists() and not force:
        logger.info("Cache hit for MRI %s — skipping preprocessing.", subject_id)
        return nib.load(str(cache_path))

    logger.info("[MRI] Preprocessing subject %s ...", subject_id)

    # 1. Load & validate
    img = load_and_validate_nifti(input_path)

    # 2. Reorient to RAS+
    img = reorient_to_ras(img)

    # 3. Resample to isotropic spacing
    img = resample_to_spacing(img, cfg.target_spacing_mm)

    # 4. Skull stripping
    img, brain_mask = skull_strip(img, cfg.skull_strip_tool, subject_id, tmp_dir)

    # 5. Bias field correction
    if cfg.bias_correction:
        img = correct_bias_field(img, brain_mask)

    # 6. MNI registration (optional)
    if cfg.mni_registration:
        logger.info("MNI registration requested but not yet implemented — skipping.")

    # 7. Intensity normalization
    img = normalize_intensity(img, cfg.normalization, brain_mask, cfg.percentile_clip)

    # 8. Crop / pad
    img = crop_or_pad(img, cfg.target_shape)

    # 9. Save to cache
    save_to_cache(img, cache_path)
    logger.info("[MRI] Done: %s → %s", subject_id, cache_path)
    return img
