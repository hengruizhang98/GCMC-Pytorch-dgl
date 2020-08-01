import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
import pandas as pd
import dgl


with open("ml-1m/ml-1m.pkl", "rb") as f:
    u_fea = pickle.load(f)
    i_fea = pickle.load(f)
    
    u_his = pickle.load(f)
    u_rat = pickle.load(f)
    
item = [] 
    
for u in u_his.keys():
    item += u_his[u]
    
item = list(set(item))

# a = []
# n = 0

# for u in u_his.keys():
#     for i in u_his[u]:
#         if i not in a:
#             a.append(i)
#             n+=1
    
    

# n_user = u_fea.shape[0]
# n_item = i_fea.shape[0]
    
# src1 = {}
# dst1 = {}
# src2 = {}
# dst2 = {}
# rating = {}
# relation = {}

# u_deg = {}
# i_deg = {}

# for u in range(n_user):
#     u_deg[u] = 0
# for i in range(n_item):
#     i_deg[i] = 0

# for i in range(5):
#     src1[i+1] = []
#     dst1[i+1] = []
#     src2[i+1] = []
#     dst2[i+1] = []

# for u in u_his.keys():
#     items = u_his[u]
#     ratings = u_rat[u]

    
#     for i in range(len(items)):
#         rat = ratings[i]
#         src1[rat].append(u)
#         dst1[rat].append(items[i])
#         u_deg[u] += 1
#         i_deg[items[i]] += 1

        
#         src2[rat].append(items[i])
#         dst2[rat].append(u)

# g_dic = {}   
 
# for i in range(1,2):
#     g_dic[('user',str(i),'item')] = (np.array(src1[i]),np.array(dst1[i]))
#     # g_dic[('item',str(i),'user')] = (np.array(src2[i]),np.array(dst2[i]))
    
        
# G = dgl.heterograph(g_dic)

# print('Node types:', G.ntypes)
# print('Edge types:', G.etypes)
# print('Canonical edge types:', G.canonical_etypes)

# print('Number of users:', G.number_of_nodes('user'))
# print('Number of items:', G.number_of_nodes('item'))

