# Create conda environment
conda create -n models_group1 python=3.7 -y
conda activate models_group1

# Install core libraries
conda install -c conda-forge matplotlib numpy pandas scikit-learn -y

pip install torch==1.9.0

# Install additional requirements
pip install torchvision reformer_pytorch

conda deactivate
