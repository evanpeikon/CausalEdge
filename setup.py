from setuptools import setup, find_packages

setup(
    name="CausalEdge",
    version="101.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "networkx",
        "statsmodels",
        "scipy",
        "scikit-learn",
        "plotly"
    ],
    author="Evan Peikon",
    description="Causal Inference for Time Series Proteomics",
    url="https://github.com/evanpeikon/CausalEdge",
)

