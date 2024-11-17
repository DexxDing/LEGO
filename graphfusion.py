import logging
import pdb
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from normal_head import NormalHead

torch.manual_seed(4869)


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN values found in {name}")
        os._exit(1)
    else:
        print(f"No NaN values found in {name}")


def check_for_inf(tensor, name):
    if torch.isinf(tensor).any():
        print(f"Inf values found in {name}")
        os._exit(1)
    else:
        print(f"No Inf values found in {name}")


logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler("ucf_training_log.txt", mode='w'), logging.StreamHandler()])


class GraphFusion(nn.Module):
    def __init__(self, m, n):
        super(GraphFusion, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.m = m
        self.n = n
        self.a = nn.Parameter(torch.normal(mean=0, std=1, size=(m, n, 1, 1, 1)))


    def take_pow(self, sim_mtx, p):

        bs_channel_dim, _, feature_dim = sim_mtx.shape
        # compute power of similarity matrix
        sim_powers = [torch.pow(sim_mtx, i) for i in range(p)]

        # set the 0th power to identity matrix
        identity_stack = torch.eye(feature_dim).unsqueeze(0).expand(bs_channel_dim, -1, -1)
        sim_powers[0] = identity_stack
        assert torch.sum(
            sim_powers[0]) == bs_channel_dim * feature_dim, f'Identity matrix sum: {torch.sum(sim_powers[0])} wrong'

        # stack powering results together
        sim_powers = torch.stack(sim_powers, dim=0)

        return sim_powers

    def single_modality_forward(self, feature, p):
        bs_channel_dim, _, feature_dim = feature.shape
        # l2 normalise
        f = F.normalize(feature, p=2, dim=2)

        # construct similarity matrix
        sim_mtx = self.relu(torch.bmm(f, f.transpose(1, 2)))

        # compute power of similarity matrix
        sim_powers = [torch.pow(sim_mtx, i) for i in range(p)]

        # set the 0th power to identity matrix
        identity_stack = torch.eye(feature_dim).unsqueeze(0).expand(bs_channel_dim, -1, -1)
        sim_powers[0] = identity_stack
        assert torch.sum(sim_powers[0]) == bs_channel_dim*feature_dim, f'Identity matrix sum: {torch.sum(sim_powers[0])} wrong'

        # stack powering results together
        sim_powers = torch.stack(sim_powers, dim=0)

        return sim_powers


    def forward(self, features):
        """
        parameters:
            features: 
                    list of features modality tensors,
                    each shaped in (bs, channel, sinppet_dim, feature_dim).
                    For training, the list should contain two tensors, 
                    For testing, the list can contain more than two tensors.
        """
        if self.training:
            # train mode
            # compute the powered similarity matrices for each modality
            modality_1, modality_2 = features
            Ga = self.single_modality_forward(modality_1, self.m)
            Gb = self.single_modality_forward(modality_2, self.n)

            # apply the fusion matrix
            fusion_mtx = self.sigmoid(self.a)
            product = Ga.unsqueeze(1) * Gb.unsqueeze(0)
            a_fused = (product * fusion_mtx).sum(dim=(0,1), keepdim=False)

            a_max = torch.max(a_fused)
            a_fused = a_fused / (a_max + 1e-10)

            return Ga, Gb, a_fused, fusion_mtx
        else:
            # test mode
            modality_1 = features[0]
            Ga = self.single_modality_forward(modality_1, self.m)

            for i in range(1,len(features)):
                modality_2 = features[i]
                # Ga = self.single_modality_forward(Ga, self.m)
                Gb = self.single_modality_forward(modality_2, self.n)

                # apply the fusion matrix
                fusion_mtx = self.sigmoid(self.a)
                product = Ga.unsqueeze(1) * Gb.unsqueeze(0)
                a_fused = (product * fusion_mtx).sum(dim=(0, 1), keepdim=False)

                a_max = torch.max(a_fused)
                a_fused = a_fused / (a_max + 1e-10)
                Ga = a_fused
                Ga = self.single_modality_forward(Ga, self.m)
            return Ga, Gb, a_fused, fusion_mtx


class Temporal(nn.Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.fc1 = nn.Linear(input_size, out_size)

    def forward(self, x):
        x = self.fc1(x)
        return x



class GraphClassifiction(nn.Module):
    def __init__(self, m, n):
        super(GraphClassifiction, self).__init__()
        #TODO: set the input size automatically or with argparse
        # self.embedding_v = Temporal(input_size=2048, out_size=32)
        # self.embedding_t = Temporal(input_size=768, out_size=32)
        self.embedding_cache = nn.ModuleList([
            Temporal(input_size=2048, out_size=32),
            # Temporal(input_size=4096, out_size=32),
            # Temporal(input_size=1024, out_size=32),
            Temporal(input_size=2176, out_size=32),
            Temporal(input_size=768, out_size=32)
        ])
        self.graphfusion = GraphFusion(m=m, n=n)
        self.normal_head = NormalHead()

#TODO:TODO:
    def get_scores(self, x, ncrops):
        outputs = self.normal_head(x)
        normal_scores = outputs

        if ncrops:
            b = normal_scores.shape[0] // ncrops
            normal_scores = normal_scores.view(b, ncrops, -1).mean(1)
        return normal_scores
#TODO:TODO:

    def forward(self, features, indices=[0, 1, 2]):
        #TODO:TODO:
        # bs, channel, sinppet_dim, feature_dim

        testlist = []
        if not self.training:
            for i in range(len(indices)):
                # pdb.set_trace()
                b, c, t, feat_v = features[i].shape
                feat = features[i].view(b*c, t, feat_v)
                testlist.append(self.embedding_cache[indices[i]](feat))
                # pdb.set_trace()
            av, at, a_fused, fusion_matrix = self.graphfusion(testlist)
            normal_scores = self.get_scores(a_fused, ncrops=c)
            return normal_scores, av, at, a_fused


        feature_1, feature_2 = features
        # pdb.set_trace()
        b, c, t, feat_v = feature_1.shape
        feat_t = feature_2.shape[3]

        # reshape the input features
        feature_1 = feature_1.view(b*c, t, feat_v)
        feature_2 = feature_2.view(b*c, t, feat_t)
        
        # embed the features
        feature_1 = self.embedding_cache[indices[0]](feature_1)
        feature_2 = self.embedding_cache[indices[1]](feature_2)
        #TODO:TODO:


        # fuse the features
        av, at, a_fused, fusion_matrix = self.graphfusion([feature_1, feature_2])

        normal_scores = self.get_scores(a_fused, ncrops=c)

        #REFINE tag begin
        # if b*c == 10:
        #     return normal_scores, av, at, a_fused

        #REFINE tag end

        fusion_resutls = dict(
            fused_graphs=a_fused,
            av=av,
            at=at
        )

        return {
            'scores': normal_scores,
            'bn_results': fusion_resutls,
            'fusion_matrix': fusion_matrix
        }




