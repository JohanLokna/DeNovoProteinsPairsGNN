# Detect installed version of PyThorch
import torch

def format_pytorch_version(version):
  return version.split('+')[0][:-1] + '0' # Must return in format X.Y.0

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

print(TORCH, CUDA, sep='+')
