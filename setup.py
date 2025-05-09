from setuptools import setup, find_packages

setup(
    name="torchcontrol",
    version="0.1.0",
    description="A parallel control system simulation and control library based on PyTorch.",
    author="Tang Longbin",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib"
    ],
    python_requires=">=3.8",
)
