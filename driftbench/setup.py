"""
Setup script for DriftBench CLI tool.

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()
else:
    long_description = "DriftBench: Cross-Stack Drift Validation for LLM Deployments"

setup(
    name="driftbench",
    version="1.0.0",
    author="Anonymous",
    author_email="anonymous@mlsys2026.submission",
    description="Cross-Stack Drift Validation for LLM Deployments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anonymous.4open.science/r/driftbench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "vllm>=0.4.0",  # Optional, for vLLM support
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "full": [
            "tensorrt>=8.6.0",  # For TensorRT-LLM support
            "sglang>=0.1.0",    # For SGLang support
        ],
    },
    entry_points={
        "console_scripts": [
            "driftbench=driftbench.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "driftbench": ["*.json", "*.yaml"],
    },
    zip_safe=False,
)
