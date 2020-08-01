import re
import os
import json
import torch
import numpy as np
import random
import pandas as pd
import dgl

from parse import parse_args

from dataset import ml1m
from model import GCMC
from evals import MAE, RMSE, MSE_loss

if __name__ == '__main__':
    
    args = parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.data_name == 'ml-1m':
        data = ml1m() 
        
    enc_G = data.enc_G
    train_dec_G = data.train_dec_G
    test_dec_G = data.test_dec_G
    u_feat = data.u_fea
    i_feat = data.i_fea
    
    u_feat = torch.Tensor(u_feat)
    i_feat = torch.Tensor(i_feat)
    
    train_labels = train_dec_G.edata['label']
    test_labels = test_dec_G.edata['label']

    config = dict()
    config['dataset'] = args.data_name
    config['n_user'] = data.n_user
    config['n_item'] = data.n_item
    
    config['u_fea_dim'] = u_feat.shape[1]
    config['i_fea_dim'] = i_feat.shape[1]
    
    config['embed_dim'] = args.embed_size 
    config['rating_values'] = data.rating_values
    config['n_relation'] = data.n_rel
    
    model = GCMC(config)   
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=5e-4)
    
    early_stop_n = 0
    model.reset_parameters()
    
    for epoch in range(args.epoch):
            
        model.train()
        optimizer.zero_grad()

        predictions = model(u_feat, i_feat, enc_G, train_dec_G)

        loss = MSE_loss(predictions, train_labels)
        
        # mae = MAE(predictions, train_labels)
        # rmse = RMSE(predictions, train_labels)

        loss.backward()
        optimizer.step()

       
        model.eval()
        
        test_pred = model(u_feat, i_feat, enc_G, test_dec_G)
        test_loss = MSE_loss(test_pred, test_labels)
        
        mae = MAE(test_pred, test_labels)
        rmse = RMSE(test_pred, test_labels)


        
        test_eval_info = {
            'epoch': epoch,
            'train_loss': loss.item(),
            'test_loss': test_loss.item(),
            'test_mae': mae,
            'test_rmse': rmse,
        }
        
        
        print(('Epoch {}, train loss {:.6f}, test loss {:.6f}, test mae {:.6f}, test rmse {:.6f}'.format(*test_eval_info.values())))

    
    
    


