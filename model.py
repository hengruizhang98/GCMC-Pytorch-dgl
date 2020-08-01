import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Parameter

from torch.autograd import Variable

import dgl
import dgl.function as fn

from layer import GCMCLayer, BiDecoder

class GCMC(nn.Module):
    
    def __init__(self, config):
        super(GCMC, self).__init__()
        rating_values = config['rating_values']
        in_dim = config['embed_dim']

        assert in_dim % len(rating_values) == 0
        msg_dim = in_dim // len(rating_values)
        out_dim = config['embed_dim']


        self.feat_tran = fea_trans(config)
         
        self.gcenc = GCMCLayer(rating_values, in_dim, msg_dim, out_dim)
        self.Linear_u = Linear(out_dim, out_dim, bias = True)
        self.Linear_i = Linear(out_dim, out_dim, bias = True)
        self.bidec = BiDecoder(rating_values, in_dim)

    def reset_parameters(self):
        self.gcenc.reset_parameters()
        self.bidec.reset_parameters()
        self.feat_tran.reset_parameters()
        self.Linear_u.reset_parameters()
        self.Linear_i.reset_parameters()

        
    def forward(self, u_feat, i_feat, enc_G, dec_G):
        u_feat, i_feat = self.feat_tran(u_feat, i_feat)

        u_emb, i_emb = self.gcenc(enc_G, u_feat, i_feat)

        u_emb = self.Linear_u(u_emb)
        i_emb = self.Linear_i(i_emb)

        pred_ratings = self.bidec(dec_G, u_emb, i_emb)

        return pred_ratings

class fea_trans(nn.Module):

    def __init__(self, config):
        super(fea_trans, self).__init__()

        u_fea_dim = config['u_fea_dim']
        i_fea_dim = config['i_fea_dim']

        if config['dataset'] == 'ml-1m':
            u_layer_dims = [u_fea_dim, u_fea_dim//10, u_fea_dim//50, config['embed_dim'] * 2, config['embed_dim'] ]
            i_layer_dims = [i_fea_dim, i_fea_dim//10, i_fea_dim//50, config['embed_dim'] * 2, config['embed_dim'] ]

        self.u_dense = dense_layer(u_layer_dims)
        self.i_dense = dense_layer(i_layer_dims)
        

    def reset_parameters(self):
        self.u_dense.reset_parameters()
        self.i_dense.reset_parameters()

    def forward(self, u_feat, i_feat):
        trans_u_feat = self.u_dense(u_feat)
        trans_i_feat = self.i_dense(i_feat)

        return trans_u_feat, trans_i_feat

class dense_layer(nn.Module):

    def __init__(self, layer_dims):
        super(dense_layer,self).__init__()
        
        self.layer_dims = layer_dims
        self.Layers = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            self.Layers.append(Linear(layer_dims[i], layer_dims[i+1], bias = True ))
    
    def reset_parameters(self):
        for layer in self.Layers:
            layer.reset_parameters()

    def forward(self, in_feat):
        assert in_feat.shape[1] == self.layer_dims[0]

        for layer in self.Layers:
            in_feat = layer(in_feat)
            in_feat = F.relu(in_feat)
        
        return in_feat


        
    

