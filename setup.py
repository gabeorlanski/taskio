import setuptools
from tio.version import VERSION

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tio",
    version=VERSION,
    author="Gabriel Orlanski",
    author_email="gabeorlanski@gmail.com",
    description="TaskIO Framework",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "test_*"],
    ),
    project_urls={
        "Documentation": "https://github.com/gabeorlanski/yamrf",
        "Source Code"  : "https://github.com/gabeorlanski/yamrf",
    },
    install_requires=[
        "transformers>=4.15.0"
        "datasets>=1.17.0",
        "overrides>=6.1.0",
        "sacrebleu>=2.0.0",
    ],
    requires_python=">=3.7",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
