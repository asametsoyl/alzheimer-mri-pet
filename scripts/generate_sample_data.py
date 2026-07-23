"""Generate realistic synthetic 3D MRI and PET NIfTI files for pipeline testing.

Creates 3 synthetic subjects in standard BIDS-like folder structure:
  data/sample_dataset/
    sub-OAS30001/
      ses-M00/
        anat/sub-OAS30001_ses-M00_T1w.nii.gz
        pet/sub-OAS30001_ses-M00_trc-fdg_pet.nii.gz
    sub-OAS30002/...
    sub-OAS30003/...
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sample_generator")


def create_synthetic_brain_pair(
    shape: tuple[int, int, int] = (96, 96, 96),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 3D MRI and PET volumes with brain-like contrast.

    Args:
        shape: 3D volume spatial dimensions (D, H, W).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mri_array, pet_array) as float32 numpy arrays.
    """
    rng = np.random.default_rng(seed)
    cz, cy, cx = [s // 2 for s in shape]
    rz, ry, rx = [s // 2.5 for s in shape]

    # Create 3D grid
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # Ellipsoid brain mask
    dist_sq = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    brain_mask = dist_sq <= 1.0

    # Inner core (subcortical / GM-like)
    core_mask = dist_sq <= 0.4

    # Outer shell (cortex-like)
    cortex_mask = (dist_sq > 0.4) & (dist_sq <= 0.85)

    # 1. MRI Volume (T1w contrast: WM high, GM medium, CSF low, skull background)
    mri = np.zeros(shape, dtype=np.float32)
    mri[brain_mask] = 0.5   # Base tissue
    mri[cortex_mask] = 0.8  # Gray matter
    mri[core_mask] = 0.9    # White matter / deep structures

    # Add Gaussian noise and bias field gradient
    bias_field = 1.0 + 0.15 * (z / shape[0] + y / shape[1])
    noise_mri = rng.normal(loc=0.0, scale=0.03, size=shape).astype(np.float32)
    mri = (mri * bias_field + noise_mri).clip(min=0.0)

    # 2. PET Volume (FDG/Amyloid tracer pattern: high cortex/subcortical uptake)
    pet = np.zeros(shape, dtype=np.float32)
    pet[cortex_mask] = 1.2  # High cortical uptake
    pet[core_mask] = 1.5    # Deep brain structure uptake
    pet[brain_mask & ~cortex_mask & ~core_mask] = 0.4  # Low background brain uptake

    noise_pet = rng.normal(loc=0.0, scale=0.05, size=shape).astype(np.float32)
    pet = (pet + noise_pet).clip(min=0.0)

    return mri, pet


def generate_sample_dataset(
    output_dir: Path,
    num_subjects: int = 3,
) -> list[Path]:
    """Generate sample BIDS dataset on disk.

    Args:
        output_dir: Target directory root.
        num_subjects: Number of sample subjects to generate.

    Returns:
        List of generated NIfTI file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])  # 1mm isotropic 4x4 matrix
    generated_files = []

    logger.info("Generating %d synthetic subjects under %s ...", num_subjects, output_dir)

    for i in range(1, num_subjects + 1):
        sub_id = f"sub-OAS3{i:04d}"
        ses_id = "ses-M00"

        anat_dir = output_dir / sub_id / ses_id / "anat"
        pet_dir = output_dir / sub_id / ses_id / "pet"
        anat_dir.mkdir(parents=True, exist_ok=True)
        pet_dir.mkdir(parents=True, exist_ok=True)

        mri_data, pet_data = create_synthetic_brain_pair(seed=42 + i)

        mri_path = anat_dir / f"{sub_id}_{ses_id}_T1w.nii.gz"
        pet_path = pet_dir / f"{sub_id}_{ses_id}_trc-fdg_pet.nii.gz"

        nib.save(nib.Nifti1Image(mri_data, affine), str(mri_path))
        nib.save(nib.Nifti1Image(pet_data, affine), str(pet_path))

        generated_files.extend([mri_path, pet_path])
        logger.info("  Created: %s and %s", mri_path.name, pet_path.name)

    logger.info("Successfully generated %d sample dataset files.", len(generated_files))
    return generated_files


if __name__ == "__main__":
    target_path = Path("data/sample_dataset")
    generate_sample_dataset(target_path, num_subjects=3)
