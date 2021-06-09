from setuptools import setup

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="porch",
    version="0.1.0",
    packages=["porch"],
    install_requires=["numpy", "tqdm"],
    description="A PyTorch Library for Scientific Machine Learning",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Raphael Leiteritz",
    license="Mozilla Public License 2.0",
    author_email="raphael.leiteritz@ipvs.uni-stuttgart.de",
    url="https://github.com/leiterrl/porch",
    keywords=[
        "porch",
        "pytorch",
        "torch",
        "machine learning",
        "physics-based",
        "physics-informed",
        "neural network",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development",
    ],
)
