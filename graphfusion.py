import logging
import pdb

import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from normal_head import NormalHead
import itertools

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
    def __init__(self, batch, m, n):
        super(GraphFusion, self).__init__()
        self.relu = nn.ReLU()
        self.m = m
        self.n = n
        self.a = nn.Parameter(torch.normal(mean=0, std=1, size=(self.m, self.n)))

    def forward(self, fv, ft):

        # set flag for testing
        test = False
        if fv.shape[0] == 10:
            test = True
            print(f'Test mode {test}')


        # l2 normalise
        f_v = F.normalize(fv, p=2, dim=2)
        f_t = F.normalize(ft, p=2, dim=2)

        # av, at
        # Cosine Similarity v
        a_v = self.relu(torch.bmm(f_v, f_v.transpose(1, 2)))
        # Cosine Similarity t
        a_t = self.relu(torch.bmm(f_t, f_t.transpose(1, 2)))


#*************************************************************************************#
        #
        # sigma = 1.0
        #
        # f_v = F.normalize(fv, p=2, dim=2)
        # f_t = F.normalize(ft, p=2, dim=2)
        #
        # dist_v = torch.sum((f_v.unsqueeze(2) - f_v.unsqueeze(1)) ** 2, dim=3)
        # dist_t = torch.sum((f_t.unsqueeze(2) - f_t.unsqueeze(1)) ** 2, dim=3)
        #
        # a_v = torch.exp(-dist_v / (2 * sigma ** 2))
        # a_t = torch.exp(-dist_t / (2 * sigma ** 2))
        #
        # a_v = F.relu(a_v)
        # a_t = F.relu(a_t)





        # compute power of av and at
        av_powers = [torch.pow(a_v, i) for i in range(self.m)]
        at_powers = [torch.pow(a_t, j) for j in range(self.n)]


        single_identity = torch.eye(32)

        single_identity = single_identity.unsqueeze(0)

        identity_stack = single_identity.repeat(fv.shape[0], 1, 1)

        av_powers[0] = identity_stack
        at_powers[0] = identity_stack



        # stacking powering results together
        av_powers = torch.stack(av_powers, dim=0)  # (m, b*c, seg, seg)
        at_powers = torch.stack(at_powers, dim=0)  # (n, b*c, seg, seg)




        #  [P, B, N, N]  0,1,2,3
        # av: [B, N, N, P], a: [P, P], at: [B, P, N, N] -> C: [B, N, N]

        # rearrange the order of dimensions to keep batch at dimension 0
        av_powers = av_powers.permute(1, 2, 3, 0)
        at_powers = at_powers.permute(1, 0, 2, 3)

        # apply fusion matrix 'a' to conduct feature fusion
        a_fused = torch.einsum('bijk,kl,blij->bij', av_powers, self.a, at_powers)  # [2b*c, seg, seg]


        # pdb.set_trace()



        # a_fused = a_v * a_t
        a_fused = self.relu(a_fused)
        # a_fused = a_t
        a_max = torch.max(a_fused)
        a_fused = a_fused / (a_max + 1e-2)
        fusion_matrix = self.a



        return a_v, a_t, a_fused, fusion_matrix   # [b*c, seg, feat(seg)]


class Temporal(nn.Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x





class GraphClassifiction(nn.Module):
    def __init__(self, batch_size, m, n):
        super(GraphClassifiction, self).__init__()
        self.embedding_v = Temporal(input_size=2048, out_size=32)
        self.embedding_t = Temporal(input_size=768, out_size=32)
        self.graphfusion = GraphFusion(batch=batch_size, m=m, n=n)
        self.normal_head = NormalHead()

        self.ratio_sample = 0.2
        self.ratio_batch = 0.4


    def get_normal_scores(self, x, ncrops=None):
        new_x = x.permute(0, 2, 1)  # [5, 512, 67]

        outputs = self.normal_head(new_x)
        normal_scores = outputs[-1]  # [5, 1, 67]
        xhs = outputs[:-1]  # xhs[0]: [5, 32, 67]  xhs[1]: [5, 16, 67]

        if ncrops:
            b = normal_scores.shape[0] // ncrops
            normal_scores = normal_scores.view(b, ncrops, -1).mean(1)

        #  normal_scores: [5, 67]
        return xhs, normal_scores


    def forward(self, v_feature, t_feature):
        b, c, t, feat_v = v_feature.shape
        feat_t = t_feature.shape[3]

        fv = v_feature.view(b*c, t, feat_v)
        ft = t_feature.view(b*c, t, feat_t)


        fv = self.embedding_v(fv)
        ft = self.embedding_t(ft)



        av, at, a_fused, fusion_matrix = self.graphfusion(fv, ft)
        # a_fused = F.normalize(a_fused, p=2, dim=-1)
        # temporal_map = self.graphfusion(fv, ft)
        #
        # fv = self.embedding_v(fv)
        # ft = self.embedding_t(ft)
        #
        # fv = F.normalize(fv, p=2, dim=-1)
        # ft = F.normalize(ft, p=2, dim=-1)
        #
        #
        # a_fused = fv * ft
        # a_fused = torch.add(fv, ft)
        # a_fused = torch.cat([fv, ft], dim=-1)
        # a_fused = ft
        # fusion_matrix = ft

        normal_feats, normal_scores = self.get_normal_scores(a_fused, ncrops=c)

        if b*c == 10:

            return normal_scores, av, at, a_fused

        bn_resutls = dict(
            fused_graphs=a_fused,
            av=av,
            at=at
        )

        return {
            'pre_normal_scores': normal_scores[0:b // 2],
            'scores': normal_scores,
            'bn_results': bn_resutls,
            'fusion_matrix': fusion_matrix
        }




