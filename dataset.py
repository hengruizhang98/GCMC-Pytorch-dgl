import pickle
import os 
import torch
import pandas as pd
import numpy as np
import argparse
import scipy.sparse as sp
import random
import dgl

def _here(*args):
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(here, *args)

class ml_1m:
    def __init__(self):
        path = _here()+ '/ml-1m/ml-1m.pkl'
        with open (path, 'rb') as f:
            u_fea = pickle.load(f)
            i_fea = pickle.load(f)
            
            u_his = pickle.load(f)
            u_rat = pickle.load(f) 

        self.n_user = u_fea.shape[0]
        self.n_item = i_fea.shape[0]
        self.rating_values = [1,2,3,4,5]
        self.n_rel = len(self.rating_values)

        self.u_fea = np.array(u_fea)
        self.i_fea = np.array(i_fea)

        train_users, train_items, test_users, test_items, u_deg, i_deg = self.train_test_split(u_his, u_rat)
        self.enc_G = self.generate_enc_graph(train_users, train_items, u_deg, i_deg)
        self.train_dec_G = self.generate_dec_graph(train_users, train_items)
        self.test_dec_G = self.generate_dec_graph(test_users, test_items)


    def train_test_split(self, u_his, u_rat):

        rel_users = {}
        rel_items = {}

        u_deg = []
        i_deg = []

        for u in range(self.n_user):
            u_deg.append(0)
        for i in range(self.n_item):
            i_deg.append(0)

        for i in range(1, self.n_rel+1):
            rel_users[i] = []
            rel_items[i] = []

        for u in u_his.keys():
            items = u_his[u]
            ratings = u_rat[u]

            for i in range(len(items)):
                rat = ratings[i]
                rel_users[rat].append(u)
                rel_items[rat].append(items[i])
                u_deg[u] += 1
                i_deg[items[i]] += 1
        
        n_per_relation = {}
        train_users = {}
        train_items = {}
        test_users = {}
        test_items = {}

        for i in range(1,self.n_rel+1):
            # train_users[i] = []
            # train_items[i] = []

            # test_users[i] = []
            # test_items[i] = []

            length = len(rel_users[i])
            n_per_relation[i] = length
            idx = list(range(n_per_relation[i]))
            random.shuffle(idx)

            train_idx = idx[:length//5 * 4]
            test_idx = idx[length//5 * 4:]

            train_users[i] = [rel_users[i][j] for j in train_idx]
            train_items[i] = [rel_items[i][j] for j in train_idx]

            test_users[i] = [rel_users[i][j] for j in test_idx]
            test_items[i] = [rel_items[i][j] for j in test_idx]
            
        return train_users, train_items, test_users, test_items, u_deg, i_deg

    def generate_enc_graph(self, rating_users, rating_items, u_deg, i_deg):
        
        g_dic = {}   
    
        for i in self.rating_values:

            g_dic[('user',str(i),'item')] = (np.array(rating_users[i]),np.array(rating_items[i]))
            g_dic[('item','rev-%s' % i,'user')] = (np.array(rating_items[i]),np.array(rating_users[i]))
            
        G = dgl.heterograph(g_dic)

        u_deg = np.sqrt(np.array(u_deg))
        i_deg = np.sqrt(np.array(i_deg))

        u_deg[np.where(u_deg == 0)] = 1
        i_deg[np.where(i_deg == 0)] = 1

        user_c = 1/u_deg
        item_c = 1/i_deg

        G.nodes['user'].data.update({'ci' : user_c})
        G.nodes['item'].data.update({'ci' : item_c})

        print('Node types:', G.ntypes)
        print('Edge types:', G.etypes)
        print('Canonical edge types:', G.canonical_etypes)

        print('Number of users:', G.number_of_nodes('user'))
        print('Number of items:', G.number_of_nodes('item'))
            
        return G

    def generate_dec_graph(self, rating_users, rating_items):
        
        u_id = []
        i_id = []

        for i in self.rating_values:
            u_id += rating_users[i]
            i_id += rating_items[i]

        ones = np.ones_like(u_id)
        user_item_ratings_coo = sp.coo_matrix(
            (ones, (u_id, i_id)),
            shape=(self.n_user, self.n_item), dtype=np.float32)
        
        return dgl.bipartite(user_item_ratings_coo, 'user', 'rate', 'item')

class ml1m:
    def __init__(self):
        path = _here()+ '/ml-1m/ml_1m.pkl'
        with open (path, 'rb') as f:
            u_fea = pickle.load(f)
            i_fea = pickle.load(f)
            
            rating_pairs = pickle.load(f)


        self.n_user = u_fea.shape[0]
        self.n_item = i_fea.shape[0]
        self.rating_values = [1,2,3,4,5]
        self.n_rel = len(self.rating_values)

        self.u_fea = np.array(u_fea)
        self.i_fea = np.array(i_fea)

        train_users, train_items, test_users, test_items, u_deg, i_deg = self.train_test_split(rating_pairs)
        self.enc_G = self.generate_enc_graph(train_users, train_items, u_deg, i_deg)
        self.train_dec_G = self.generate_dec_graph(train_users, train_items)
        self.test_dec_G = self.generate_dec_graph(test_users, test_items)

    def train_test_split(self, rating_pairs):

        rel_users = {}
        rel_items = {}

        u_deg = []
        i_deg = []

        for u in range(self.n_user):
            u_deg.append(0)
        for i in range(self.n_item):
            i_deg.append(0)


        users = rating_pairs[0]
        items = rating_pairs[1]
        ratings = rating_pairs[2]

        for i in range(1, self.n_rel+1):
            rel_users[i] = []
            rel_items[i] = []

        
        for u,i,r in zip(users,items,ratings):
            rel_users[r].append(u)
            rel_items[r].append(i)
            
            u_deg[u] += 1
            i_deg[i] += 1

        
        # for u in u_his.keys():
        #     items = u_his[u]
        #     ratings = u_rat[u]

        #     for i in range(len(items)):
        #         rat = ratings[i]
        #         rel_users[rat].append(u)
        #         rel_items[rat].append(items[i])
        #         u_deg[u] += 1
        #         i_deg[items[i]] += 1
        
        n_per_relation = {}
        train_users = {}
        train_items = {}
        test_users = {}
        test_items = {}

        for i in range(1,self.n_rel+1):
            # train_users[i] = []
            # train_items[i] = []

            # test_users[i] = []
            # test_items[i] = []

            length = len(rel_users[i])
            n_per_relation[i] = length
            idx = list(range(n_per_relation[i]))
            random.shuffle(idx)

            train_idx = idx[:length//5 * 4]
            test_idx = idx[length//5 * 4:]

            train_users[i] = [rel_users[i][j] for j in train_idx]
            train_items[i] = [rel_items[i][j] for j in train_idx]

            test_users[i] = [rel_users[i][j] for j in test_idx]
            test_items[i] = [rel_items[i][j] for j in test_idx]
            
        return train_users, train_items, test_users, test_items, u_deg, i_deg

    def generate_enc_graph(self, rating_users, rating_items, u_deg, i_deg):
        
        g_dic = {}   
    
        for i in self.rating_values:

            g_dic[('user',str(i),'item')] = (np.array(rating_users[i]),np.array(rating_items[i]))
            g_dic[('item','rev-%s' % i,'user')] = (np.array(rating_items[i]),np.array(rating_users[i]))
            
        G = dgl.heterograph(g_dic)

        u_deg = np.sqrt(np.array(u_deg))
        i_deg = np.sqrt(np.array(i_deg))

        u_deg[np.where(u_deg == 0)] = 1
        i_deg[np.where(i_deg == 0)] = 1

        user_c = torch.Tensor(1/u_deg).unsqueeze(1)
        item_c = torch.Tensor(1/i_deg).unsqueeze(1)

        G.nodes['user'].data.update({'sqrt_deg' : user_c})
        G.nodes['item'].data.update({'sqrt_deg' : item_c})

        print('Node types:', G.ntypes)
        print('Edge types:', G.etypes)
        print('Canonical edge types:', G.canonical_etypes)

        print('Number of users:', G.number_of_nodes('user'))
        print('Number of items:', G.number_of_nodes('item'))
            
        return G

    def generate_dec_graph(self, rating_users, rating_items):
        
        u_id = []
        i_id = []
        r = []
        
        for i in self.rating_values:
            u_id += rating_users[i]
            i_id += rating_items[i]
            r += [i for j in range(len(rating_users[i]))]
            
        # print(u_id[0], i_id[0])
        r = torch.Tensor(r)
        # r = torch.IntTensor(r)

        ones = np.ones_like(u_id)
        user_item_ratings_coo = sp.coo_matrix(
            (ones, (u_id, i_id)),
            shape=(self.n_user, self.n_item), dtype=np.float32)
        
        G = dgl.bipartite(user_item_ratings_coo, 'user', 'rate', 'item')
        G.edata['label'] = r
        
        # print(G.find_edges(0))
        return G


if __name__ == '__main__':
    data = ml1m()