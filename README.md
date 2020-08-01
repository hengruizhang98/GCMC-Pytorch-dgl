#  GCMC-Pytorch-dgl

## Introduction

A Pytorch implementation of GCMC model with Deep Graph Library (DGL). The user-item bipartite graph is built using dgl-Heterogeneous Graph.

## Paper

The gcmc model is proposed by the [paper](https://arxiv.org/abs/1706.02263) below:

```
@article{DBLP:journals/corr/BergKW17,
  author    = {Rianne van den Berg and
               Thomas N. Kipf and
               Max Welling},
  title     = {Graph Convolutional Matrix Completion},
  journal   = {CoRR},
  volume    = {abs/1706.02263},
  year      = {2017},
  url       = {http://arxiv.org/abs/1706.02263},
}
```

## Run the code

Run the following command in the terminal:

`python main.py --data_name ml-1m --gpu 0 --epoch 100 --embed_size 20 --lr 1e-2 ` 

## Meaning of the arguments

```
--lr: learning rate
--gpu: gpu id
--epoch: number of training epoches
--embed_size: size of the hidden representations of nodes, should be able to be divided by  the number of possible rating values(5 in ml-1m).
```

There are also several optional arguments for this model, read parse.py for details.



