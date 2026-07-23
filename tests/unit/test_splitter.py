"""Unit tests for subject-level splitter and leakage verification."""

import sys
sys.path.insert(0, "src")

import pytest
from data.adapters.base import SubjectRecord
from data.splitter import split_subjects, verify_no_leakage
from pathlib import Path


def make_record(subject_id: str, visit_id: str = "ses-baseline") -> SubjectRecord:
    return SubjectRecord(
        subject_id=subject_id,
        visit_id=visit_id,
        mri_path=Path(f"/fake/{subject_id}/{visit_id}/mri.nii.gz"),
        pet_path=Path(f"/fake/{subject_id}/{visit_id}/pet.nii.gz"),
        dataset_name="test",
    )


class TestSplitter:
    def _make_records(self, n_subjects=20, visits_per_subject=1):
        records = []
        for i in range(n_subjects):
            sid = f"sub-{i:04d}"
            for j in range(visits_per_subject):
                records.append(make_record(sid, f"ses-M{j:03d}"))
        return records

    def test_basic_split(self):
        records = self._make_records(20)
        split = split_subjects(records, 0.7, 0.15, 0.15, seed=42)
        assert len(split.train) > 0
        assert len(split.val) > 0
        assert len(split.test) > 0

    def test_no_overlap_single_visit(self):
        records = self._make_records(30)
        split = split_subjects(records)
        assert len(split.train_subjects & split.val_subjects) == 0
        assert len(split.train_subjects & split.test_subjects) == 0
        assert len(split.val_subjects & split.test_subjects) == 0

    def test_no_overlap_multi_visit(self):
        """Multi-visit subjects must stay in the same split."""
        records = self._make_records(20, visits_per_subject=3)
        split = split_subjects(records)
        assert len(split.train_subjects & split.val_subjects) == 0
        assert len(split.train_subjects & split.test_subjects) == 0

    def test_ratios_sum(self):
        records = self._make_records(100)
        split = split_subjects(records, 0.7, 0.15, 0.15)
        n_total = len(split.train_subjects | split.val_subjects | split.test_subjects)
        assert n_total == 100

    def test_too_few_subjects_raises(self):
        records = self._make_records(2)
        with pytest.raises(ValueError, match="at least 3"):
            split_subjects(records)


class TestLeakageVerification:
    def test_no_leakage_passes(self):
        records = [make_record(f"sub-{i:04d}") for i in range(30)]
        split = split_subjects(records)
        verify_no_leakage(split)  # should not raise

    def test_injected_leakage_raises(self):
        """Manually inject a leaking record and verify it raises."""
        from data.splitter import DataSplit  # noqa: PLC0415
        shared = make_record("sub-SHARED")
        split = DataSplit(
            train=[make_record("sub-A"), shared],
            val=[make_record("sub-B"), shared],  # same subject in val!
            test=[make_record("sub-C")],
        )
        with pytest.raises(RuntimeError, match="LEAKAGE"):
            verify_no_leakage(split)
