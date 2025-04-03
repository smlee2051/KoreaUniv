conda create -n models_group1_env python=3.7 -y
conda activate models_group1_env

conda install -c conda-forge matplotlib numpy pandas scikit-learn -y

pip install torch==1.9.0

pip install torchvision reformer_pytorch

conda deactivate
