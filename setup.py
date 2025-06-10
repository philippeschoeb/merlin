"""
Setup script for Merlin - Photonic Quantum Neural Networks.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    scmver=True,
    setup_requires=["scmver"],
    long_description=long_description,
    package_dir={"": "."},
    packages=find_packages(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "pytest-benchmark",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.0",
        ],
        "examples": [
            "matplotlib>=3.5.0",
            "scikit-learn>=1.0.0",
            "jupyter>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
