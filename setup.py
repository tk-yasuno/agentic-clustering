from setuptools import setup, find_packages

setup(
    name="agentic-clustering",
    version="0.5.0",
    description="Self-improving clustering for bridge maintenance prioritization",
    author="Takayuki Yasuno",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.1",
        "hdbscan>=0.8.27",
        "umap-learn>=0.5.1",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "geopy>=2.2.0",
        "shapely>=1.8.0",
        "geopandas>=0.10.0",
    ],
)
