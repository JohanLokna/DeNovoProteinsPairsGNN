from setuptools import find_packages, setup

def read_md(file):
    with open(file) as fin:
        return fin.read()


setup(
    name="DeNovoProteinsPairsGNN",
    version="0.1.1",
    description="Create a framework for de novo generation of interacting protein pairs using graphical neural networks.",
    long_description=read_md("README.md"),
    author="Johan Lokna",
    author_email="jlokna@math.ethz.ch",
    url="https://github.com/JohanLokna/DeNovoProteinsPairsGNN",
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ]
)
