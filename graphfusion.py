import logging
import pdb

import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import option
from normal_head import NormalHead
import itertools


torch.manual_seed(4869)


args = option.parser.parse_args()

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



class Fusion(nn.Module):
    def __init__(self, d_1, d_2):
        super(Fusion, self).__init__()
        self.fc1 = nn.Linear(d_1, 1)
        self.fc2 = nn.Linear(d_2, 1)

    def forward(self, x):
        x = x.permute(1, 2, 3, 4, 0)
        x = self.fc1(x).squeeze(-1)
        x = x.permute(1, 2, 3, 0)
        x = self.fc2(x).squeeze(-1)
        return x

class GraphFusion(nn.Module):
    def __init__(self, batch, m, n):
        super(GraphFusion, self).__init__()
        self.relu = nn.ReLU()
        self.m = m
        self.n = n
        self.a = nn.Parameter(torch.normal(mean=0, std=1, size=(self.m, self.n, 1, 1, 1)))
        self.sigmoid = nn.Sigmoid()
        # self.a = nn.Parameter(torch.rand(self.m, self.n))
        # self.a = nn.Parameter(torch.abs(torch.normal(mean=0, std=1, size=(self.m, self.n))))
        if args.go == 'fc':
            self.fusion = Fusion(self.m, self.n)






    def forward(self, fv, ft):

        fusion_operator = self.sigmoid(self.a)

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
        # sigma = 0.92 / 4
        #
        # f_v = F.normalize(fv, p=2, dim=2)
        # f_t = F.normalize(ft, p=2, dim=2)
        #
        # # pdb.set_trace()
        #
        # dist_v = torch.sum((f_v.unsqueeze(2) - f_v.unsqueeze(1)) ** 2, dim=3)
        # dist_t = torch.sum((f_t.unsqueeze(2) - f_t.unsqueeze(1)) ** 2, dim=3)
        #
        #
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
        # av_powers = av_powers.permute(1, 2, 3, 0)
        # at_powers = at_powers.permute(1, 0, 2, 3)



        pow_product = av_powers.unsqueeze(1) * at_powers.unsqueeze(0)   # [m, n, b, seg, seg]

        if args.go == 'fc':
            a_fused = self.fusion(pow_product)
            a_max = torch.max(a_fused)
            a_fused = a_fused / (a_max + 1e-10)
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix  # [b*c, seg, feat(seg)]

        if args.modal == 'fv':
            a_fused = f_v
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix

        if args.modal == 'ft':
            a_fused = f_t
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix

        if args.modal == 'fv+ft':
            a_fused = f_v + f_t
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix

        if args.modal == 'fvxft':
            a_fused = f_v * f_t
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix


        if args.modal == 'fvcft':
            a_fused = torch.cat([f_v, f_t], dim=-1)
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix




        if args.modal == 'av':
            a_fused = a_v
            a_max = torch.max(a_fused)
            a_fused = a_fused / (a_max + 1e-10)
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix  # [b*c, seg, feat(seg)]

        if args.modal == 'at':
            a_fused = a_t
            a_max = torch.max(a_fused)
            a_fused = a_fused / (a_max + 1e-10)
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix  # [b*c, seg, feat(seg)]

        if args.modal == 'av+at':
            a_fused = a_v + a_t
            a_max = torch.max(a_fused)
            a_fused = a_fused / (a_max + 1e-10)
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix  # [b*c, seg, feat(seg)]

        if args.modal == 'avxat':
            a_fused = a_v * a_t
            a_max = torch.max(a_fused)
            a_fused = a_fused / (a_max + 1e-10)
            fusion_matrix = a_fused
            return a_v, a_t, a_fused, fusion_matrix  # [b*c, seg, feat(seg)]

        if args.modal == 'avcat':
            a_fused = torch.cat([a_v, a_t], dim=-1)
            a_max = torch.max(a_fused)
            a_fused = a_fused / (a_max + 1e-10)
            fusion_matrix = a_fused

            return a_v, a_t, a_fused, fusion_matrix  # [b*c, seg, feat(seg)]


        a_fused = (pow_product * fusion_operator).sum(dim=(0, 1), keepdim=False)

        # apply fusion matrix 'a' to conduct feature fusion
        # a_fused = torch.einsum('bijk,kl,blij->bij', av_powers, self.a, at_powers)  # [2b*c, seg, seg]

        # a_fused = F.sigmoid(self.a) * a_v * (1-F.sigmoid(self.a)) * a_t


        # pdb.set_trace()



        # a_fused = a_v * a_t
        # a_fused = self.sigmoid(a_fused)
        # a_fused = self.relu(a_fused)
        # a_fused = a_v
        a_max = torch.max(a_fused)
        a_fused = a_fused / (a_max + 1e-10)
        fusion_matrix = self.a





        return a_v, a_t, a_fused, fusion_matrix   # [b*c, seg, feat(seg)]


# class Temporal(nn.Module):
#     def __init__(self, input_size, out_size):
#         super(Temporal, self).__init__()
#         self.conv_1 = nn.Sequential(
#             nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
#                     stride=1, padding=1),
#             nn.ReLU(),
#         )
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.conv_1(x)
#         x = x.permute(0, 2, 1)
#         return x
    


class Temporal(nn.Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.fc1 = nn.Linear(input_size, out_size)

    def forward(self, x):
        x = self.fc1(x)
        return x






class GraphClassifiction(nn.Module):
    def __init__(self, batch_size, m, n):
        super(GraphClassifiction, self).__init__()
        self.embedding_v = Temporal(input_size=2048, out_size=32)
        self.embedding_t = Temporal(input_size=768, out_size=32)
        self.graphfusion = GraphFusion(batch=batch_size, m=m, n=n)
        self.normal_head = NormalHead(input_dim1=32)
        if args.modal == 'avcat' or args.modal == 'fvcft':
            self.normal_head = NormalHead(input_dim1=64)


    def get_scores(self, x, ncrops=None):


        scores = self.normal_head(x)

        if ncrops:
            b = scores.shape[0] // ncrops
            scores = scores.view(b, ncrops, -1).mean(1)

        #  scores: [5, 67]
        return scores


    def forward(self, v_feature, t_feature):
        b, c, t, feat_v = v_feature.shape
        feat_t = t_feature.shape[3]

        fv = v_feature.view(b*c, t, feat_v)
        ft = t_feature.view(b*c, t, feat_t)


        fv = self.embedding_v(fv)
        ft = self.embedding_t(ft)

        av, at, a_fused, fusion_matrix = self.graphfusion(fv, ft)

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
        # pdb.set_trace()
        # a_fused = fv
        # av = a_fused
        # at = a_fused
        # fusion_matrix = a_fused


        scores = self.get_scores(a_fused, ncrops=c)

        if b*c == 10:
            return scores, av, at, a_fused

        fusion_resutls = dict(
            fused_graphs=a_fused,
            av=av,
            at=at
        )

        return {
            'scores': scores,
            'bn_results': fusion_resutls,
            'fusion_matrix': fusion_matrix
        }




