conda create -n models_group3 python=3.11 -y
conda activate models_group3

pip install torch==2.2.2
pip install accelerate==0.28.0
pip install einops==0.7.0
pip install matplotlib==3.7.0
pip install numpy==1.23.5
pip install pandas==1.5.3
pip install scikit_learn==1.2.2
pip install scipy==1.12.0
pip install tqdm==4.65.0
pip install peft==0.4.0
pip install transformers==4.31.0
pip install deepspeed==0.14.0
pip install sentencepiece==0.2.0

conda deactivate
