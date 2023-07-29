sudo apt-get install conda
conda create -n iliuhin python=3.9.6
source activate iliuhin

conda install pip
pip install torch==1.13.1
pip install transformers==4.30.0
pip install sentence-transformers
pip install datasets
pip install huggingface-hub
pip install xformers

