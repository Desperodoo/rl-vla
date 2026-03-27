#!/usr/bin/env python3
"""
Setup script for the Robometer package.
"""

from setuptools import find_packages, setup

# Read the README file
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="robometer",
    version="0.0.1",
    author="Anthony Liang",
    author_email="aliang80@usc.edu",
    description="Robometer: Scaling General-purpose Robotic Reward Models via Trajectory Comparisons",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robometer/robometer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "robometer-train=train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
