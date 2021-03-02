pip install --upgrade pip

# Install normal requirements
pip install -r ./requirements.txt

# Install torch geometric requirements
version=$(python get_pytorch_geometric_version.py)
pip install torch-scatter==1.4.3 -f https://pytorch-geometric.com/whl/torch-${version}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${version}.html
pip install torch-geometric
