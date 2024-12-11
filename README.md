# Supply_link_prediction

 <!-- Dependencies -->
Create a Conda virtual environment and install all the necessary packages

'''
conda create --name ES-NHPC python=3.9
conda activate ES-NHPC
'''

'''
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install tqdm
pip install torch_geometric
pip install pandas
pip install scikit-learn
pip install modelscope
pip install vllm
pip install pytorch_lightning
pip install transformers
'''

<!-- Datasets -->
We use `bert` and `qwen` for text feature extraction and address search embedding, and we utilize adjacency matrix information, which can be found in the `dataset`.

<!-- Usage -->
To reproduce the code, please focus on the `model.py` and `NHPE.py` files. We use the code in the `embedding` directory to process and obtain data from the `dataset` folder. You only need to replace the specified paths in `NHPE.py` and `model.py` with your corresponding paths.

- `embedding_path_qwen`: the embedding of qwen
- `embedding_path_bert`: the embedding of bert
- `adj17_path`: the adjacency matrix in 2023-06-30
- `a15_path`: the adjacency matrix in 2022-06-30
- `industry_path`: the information of industry_chain
- `node_index`: the index of nodes
- `location_path_qwen`: the embedding of location
- `location_path_bert`: the embedding of location
- `edge_status_json`: counting of node pairs
