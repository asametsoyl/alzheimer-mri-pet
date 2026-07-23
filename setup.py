"""Setup for the alzheimer_mri_pet package."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="alzheimer_mri_pet",
    version="0.1.0",
    description="Publication-quality MRI → PET synthesis pipeline for Alzheimer's Disease",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Research Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "pytest-cov", "ruff"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
