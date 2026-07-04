# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", errors="ignore", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metabci",
    version="0.1.2",
    author="TBC-TJU",
    author_email="TBC_TJU_2022@163.com",
    description="A Library of Datasets, Algorithms, \
        and Experiments workflow for Brain-Computer Interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    python_requires=">=3.8,<3.9",
    install_requires=[
        "autograd==1.7.0",
        "h5py==3.11.0",
        "joblib==1.4.2",
        "mat73==0.65",
        "matplotlib==3.7.5",
        "mne==1.6.1",
        "numpy==1.23.5",
        "pandas==2.0.3",
        "pooch==1.8.2",
        "PsychoPy==2020.1.2",
        "py7zr==0.22.0",
        "pyglet==1.4.11",
        "pylsl==1.16.2",
        "pymanopt==0.2.5",
        "pyserial==3.5",
        "requests[socks]==2.32.4",
        "robomaster==0.1.1.68",
        "scikit-learn==1.3.2",
        "scipy==1.10.1",
        "skorch==1.0.0",
        "sympy==1.13.3",
        "torch==2.4.1",
        "tqdm==4.67.3",
        "wxPython==4.2.2; sys_platform == 'win32'",
    ],
    extras_require={
        "build": [
            "setuptools==70.3.0",
            "twine==6.1.0",
            "wheel==0.44.0",
        ],
        "dev": [
            "coverage==7.6.1",
            "flake8==7.1.2",
            "mypy==1.14.1",
            "pytest==8.3.5",
        ],
        "docs": [
            "sphinxcontrib-napoleon==0.7",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
)
