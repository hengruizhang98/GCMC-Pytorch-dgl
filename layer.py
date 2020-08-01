import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Parameter

from torch.autograd import Variable

import dgl
import dgl.function as fn

class GCMCLayer(nn.Module):
    def __init__(self,
                 rating_vals,
                 in_dim,
                 msg_dim,
                 out_dim,
                 dropout_rate=0.0,
                 agg='stack',  # or 'sum'
                 agg_act= True,
                 share_user_item_param=False):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param

        self.agg_act = agg_act
        
            
        self.dropout = nn.Dropout(dropout_rate)

        self.conv_u = torch.nn.ModuleList()
        self.conv_i = torch.nn.ModuleList()
        
        if share_user_item_param:
            for i in range(len(rating_vals)):
                self.conv_u.append(Linear(in_dim, msg_dim, bias = False))
                self.conv_i.append(self.conv_u[i])
        else:
            for i in range(len(rating_vals)):
                self.conv_u.append(Linear(in_dim, msg_dim, bias = False))
                self.conv_i.append(Linear(in_dim, msg_dim, bias = False))

    def reset_parameters(self):          
        if self.share_user_item_param:
            for lin in self.conv_u:
                lin.reset_parameters()
        else:
            for lin in self.conv_u:
                lin.reset_parameters()
            for lin in self.conv_i:
                lin.reset_parameters()

    def forward(self, graph, ufeat=None, ifeat=None):

        num_u = graph.number_of_nodes('user')
        num_i = graph.number_of_nodes('item')
        
        funcs = {}
        for i, rating in enumerate(self.rating_vals):
            rating = str(rating)
            # W_r * x
            x_u = self.conv_u[i](ufeat)
            x_i = self.conv_i[i](ifeat)
    
            # left norm and dropout
            x_u = x_u * self.dropout(graph.nodes['user'].data['sqrt_deg'])
            x_i = x_i * self.dropout(graph.nodes['item'].data['sqrt_deg'])

            graph.nodes['user'].data['h%d' % i] = x_u
            graph.nodes['item'].data['h%d' % i] = x_i

            funcs[rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
            funcs['rev-%s' % rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))

        # message passing
        graph.multi_update_all(funcs, self.agg)

        ufeat = graph.nodes['user'].data.pop('h').reshape((num_u, -1))
        ifeat = graph.nodes['item'].data.pop('h').reshape((num_i, -1))

        # right norm
        ufeat = ufeat * graph.nodes['user'].data['sqrt_deg']
        ifeat = ifeat * graph.nodes['item'].data['sqrt_deg']

        # non-linear
       
        ufeat = F.relu(ufeat)
        ifeat = F.relu(ifeat)

        return ufeat, ifeat

class BiDecoder(nn.Module):

    def __init__(self,
                 rating_vals,
                 in_dim,
                 num_basis_functions=2,
                 dropout_rate=0.0):
        super(BiDecoder, self).__init__()

        self.rating_vals = rating_vals
        self._num_basis_functions = num_basis_functions
        self.dropout = nn.Dropout(dropout_rate)


        self.Ps = []
        
        for i in range(len(self.rating_vals)):
            self.Ps.append(Parameter(torch.Tensor(in_dim, in_dim)))
            torch.nn.init.xavier_uniform_(self.Ps[i])

        self.rate_out = Linear(in_dim * len(rating_vals), len(rating_vals), bias = True)
    
    def reset_parameters(self):
        
        self.rate_out.reset_parameters

    def forward(self, graph, ufeat, ifeat):
        """Forward function.
        Parameters
        ----------
        graph : DGLHeteroGraph
            "Flattened" user-movie graph with only one edge type.
        ufeat : torch.Tensor
            User embeddings. Shape: (|V_u|, D)
        ifeat : torch.Tensor
            Movie embeddings. Shape: (|V_m|, D)
        Returns
        -------
        torch.Tensor
            Predicting scores for each user-movie edge.
        """
        graph = graph.local_var()
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        graph.nodes['item'].data['h'] = ifeat
        basis_out = []
        for i in range(len(self.rating_vals)):
            graph.nodes['user'].data['h'] = torch.mm(ufeat, self.Ps[i])
            graph.apply_edges(fn.u_dot_v('h', 'h', 'sr'))
            # basis_out.append(graph.edata['sr'].expand_dims(1))
            basis_out.append(torch.unsqueeze(graph.edata['sr'],1))

        out = torch.cat(basis_out, dim=1)
        
        out = F.softmax(out, dim = 1)
        possible_ratings = torch.Tensor(self.rating_vals)

        ratings = torch.sum(out*possible_ratings, dim =1)
        return ratings
    
# def dot_or_identity(A, B):
#     # if A is None, treat as identity matrix
#     if A is None:
#         return B
#     else:
#         return torch.mm(A, B)