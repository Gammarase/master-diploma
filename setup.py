"""Package setup for the Disinformation Detection System."""

from setuptools import find_packages, setup

_INSTALL_REQUIRES = [
    "spacy>=3.7.0",
    "langdetect>=1.0.9",
    "transformers>=4.40.0",
    "sentence-transformers>=2.7.0",
    "torch>=2.2.0",
    "pinecone>=3.2.0",
    "httpx>=0.27.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "beautifulsoup4>=4.12.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "PyYAML>=6.0.1",
]

_DEV_REQUIRES = [
    "pytest>=8.2.0",
    "pytest-cov>=5.0.0",
    "pytest-mock>=3.14.0",
    "black>=24.4.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]

setup(
    name="disinformation-detection",
    version="0.1.0",
    description=(
        "Modular Python pipeline for detecting disinformation using "
        "Pinecone vector search, NLI models, and Ollama LLMs."
    ),
    author="Krivytskyi Bohdan",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*", "data*"]),
    install_requires=_INSTALL_REQUIRES,
    extras_require={"dev": _DEV_REQUIRES},
    entry_points={
        "console_scripts": [
            "disinfo-index=data.loaders.ru22fact_loader:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
