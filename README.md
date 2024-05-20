# EMGNN

This repo supplements our [paper](https://openreview.net/forum?id=VyfEv6EjKR) published in ICML-24. We explored employing a distribution of parametrized graphs for training a GNN in an Expectation Maximization (EM) framework. Through a probabilistic framework, we handle the uncertainty in graph structures stemming from various sources. Our approach enables the model to handle multiple graphs.

## Data (Heterogeneous Graphs)

Download datasets (DBLP, ACM, IMDB) from this [link](https://drive.google.com/file/d/1Nx74tgz_-BDlqaFO75eQG6IkndzI92j4/view?usp=sharing) and extract data.zip into `data` folder. (reference: [GTN](https://github.com/seongjunyun/Graph_Transformer_Networks#running-the-code))

Alternatively, unzip `data.zip` into `data` folder.

## Experiment Command

The result of each dataset is the corresponding text files `out_ACM.txt`, `out_DBLP.txt`, `out_IMDB.txt`, and `out_CORA.txt`.

### IMDB
`python main_batch_gcn.py --usedataset=IMDB --gamma=4 --alpha=0.2 --T=15 --Tprime=25`

### DBLP
`python main_batch_gcn.py --usedataset=DBLP --gamma=4 --alpha=0.2 --T=15 --Tprime=25 --patience=200 --lr=0.003`

### ACM
`python main_batch_gcn.py --usedataset=ACM --gamma=2 --alpha=0.2 --T=15 --Tprime=25 --patience=150 --lr=0.005 --layer=2`

### CORA
`python main_homogeneous2.py --n-layers=2 --n-epochs=300 --gpu=1 --label_n_per_class=20 --dropout=0.8 --dataset=cora --self-loop `
