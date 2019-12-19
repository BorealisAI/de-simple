# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset

class DE_DistMult(torch.nn.Module):
    '''
    Implements the DE_DistMult model in https://arxiv.org/abs/1907.03143
    '''
    def __init__(self, dataset, params):

        super(DE_DistMult, self).__init__()
        self.dataset = dataset
        self.params = params

        # Creating static embeddings.
        self.ent_embs      = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        
        # Creating and initializing the temporal embeddings for the entities 
        self.create_time_embedds()
        
        # Setting the non-linearity to be used for temporal part of the embedding
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)

        
    def create_time_embedds(self):
        
        # frequency embeddings for the entities
        self.m_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        # phi embeddings for the entities
        self.m_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        # amplitude embeddings for the entities
        self.m_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)
        
            
    def get_time_embedd(self, entities, year, month, day):

        y = self.y_amp(entities)*self.time_nl(self.y_freq(entities)*year + self.y_phi(entities))
        m = self.m_amp(entities)*self.time_nl(self.m_freq(entities)*month + self.m_phi(entities))
        d = self.d_amp(entities)*self.time_nl(self.d_freq(entities)*day + self.d_phi(entities))

        return y+m+d

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        
        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)        
        h_t = self.get_time_embedd(heads, years, months, days)
        t_t = self.get_time_embedd(tails, years, months, days)
        
        h = torch.cat((h,h_t), 1)
        t = torch.cat((t,t_t), 1)
        return h,r,t
        
    def forward(self, heads, rels, tails, years, months, days):
        h_embs, r_embs, t_embs = self.getEmbeddings(heads, rels, tails, years, months, days)
        
        scores = (h_embs * r_embs) * t_embs
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        
        return scores
        