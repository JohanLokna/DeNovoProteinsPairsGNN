import sys
import os

pipPath = sys.argv[1]

import torch
TORCH, CUDA = torch.__version__.split("+")

# Install correct versions of Pytorch Geometric, Scatter and Sparse
TORCH = TORCH[:-1] + "0"
os.system(pipPath + " uninstall torch-scatter torch-sparse torch-geometric --yes")
os.system(pipPath + " install torch-scatter -f https://pytorch-geometric.com/whl/torch-{}+{}.html --no-cache-dir".format(TORCH, CUDA))
os.system(pipPath + " install torch-sparse -f https://pytorch-geometric.com/whl/torch-{}+{}.html --no-cache-dir".format(TORCH, CUDA))
os.system(pipPath + " install torch-geometric==1.4.3 --upgrade --no-cache-dir")

# Install ProteinPairsGenerator
os.system(pipPath + " install git+https://github.com/JohanLokna/DeNovoProteinsPairsGNN.git@modular --upgrade")

# Install ANARCI
os.system("conda install -c bioconda hmmer")
os.system(pipPath + " install git+https://github.com/oxpig/ANARCI.git")

import anarci
import pathlib
anarciPath = pathlib.Path(anarci.__file__).parent
anarciPath.joinpath("dat/").mkdir(exist_ok=True)
anarciPath.joinpath("dat/HMMs/").mkdir(exist_ok=True)
os.system("git clone https://github.com/oxpig/ANARCI.git")
os.system("cp ANARCI/lib/python/anarci/dat/HMMs/* {}/dat/HMMs/".format(str(anarciPath)))
os.system("rm ANARCI/ -rf")
