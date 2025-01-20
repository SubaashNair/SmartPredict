
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README and CHANGELOG files
this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
changelog_path = this_directory / "CHANGELOG.md"

long_description = ""
if readme_path.exists():
    long_description += readme_path.read_text(encoding='utf-8')
if changelog_path.exists():
    long_description += "\n\n" + changelog_path.read_text(encoding='utf-8')

setup(
    name="smartpredict",
    version="0.8.0",  # Update the version number
    description="An advanced machine learning library for effortless model training, evaluation, and selection.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Subashanan Nair",
    author_email="subaashnair12@gmail.com",
    url="https://github.com/SubaashNair/SmartPredict",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "numpy",
        "pandas",
        "shap",
        "optuna",
        "xgboost",
        "lightgbm",
        "catboost",
        "tensorflow",
        "torch"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
