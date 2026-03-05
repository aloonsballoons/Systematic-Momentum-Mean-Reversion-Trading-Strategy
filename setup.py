from setuptools import setup, find_packages

setup(
    name="quant-momentum-mr",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "statsmodels>=0.14",
        "matplotlib>=3.7",
        "plotly>=5.14",
        "yfinance>=0.2.18",
        "PyYAML>=6.0",
        "tabulate>=0.9",
        "tqdm>=4.65",
        "loguru>=0.7",
        "pyarrow>=12.0",
        "pytest>=7.3",
    ],
)
