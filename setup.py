from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smartpredict",
    version="0.1.2",
    author="Subaash Nair",
    author_email="your.email@example.com",
    description="An advanced machine learning library for model training and selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SubaashNair/smartpredict",
    project_urls={
        "Bug Tracker": "https://github.com/SubaashNair/smartpredict/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "optuna>=2.0.0",
        "shap>=0.39.0",
    ],
)