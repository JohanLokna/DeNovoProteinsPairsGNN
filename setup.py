from setuptools import find_packages, setup

def read_md(file):
    with open(file) as fin:
        return fin.read()

# Requrements affacting multiple dependencies
TORCH="1.7.1"
CUDA="cu101"

setup(
    name="DeNovoProteinsPairsGNN",
    version="0.0.1",
    description="Create a framework for de novo generation of interacting protein pairs using graphical neural networks.",
    long_description=read_md("README.md"),
    author="Johan Lokna",
    author_email="jlokna@math.ethz.ch",
    url="https://github.com/JohanLokna/DeNovoProteinsPairsGNN",
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
      "torch=={}+{}".format(TORCH, CUDA),
      "torch-scatter",
      "torch-sparse",
      "torch-geometric==1.4.3"
    ],
    dependency_links=[
      "https://pytorch-geometric.com/whl/torch-${}+{}.html".format(TORCH[:-1] ' "0', CUDA) # Torch specification must be X.X.0
    ]
)
