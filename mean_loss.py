import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanLoss(nn.Module):
    def __init__(self, k, threshold):
        super(MeanLoss, self).__init__()
        self.k = k
        self.threshold = threshold


    def forward(self, results):
        features = results['bn_results']['fused_graphs']

        b, c, t = features.size()

        g_nor = features[0:b // 2]
        g_abn = features[b // 2:]

        mean_nors_nor = []
        mean_nors_abn = []
        for i in range(b // 2):
            a_nor = g_nor[i]
            a_abn = g_abn[i]

            mask_nor = a_nor >= self.threshold
            mask_abn = a_abn >= self.threshold

            a_nor = a_nor * mask_nor.float()
            a_abn = a_abn * mask_abn.float()

            # pdb.set_trace()

            deg_nor = torch.sum(a_nor, dim=-1)
            topk_abn = torch.topk(a_abn, k=self.k, dim=-1)[0]
            deg_abn = torch.sum(topk_abn, dim=-1)

            mean_nor = torch.mean(deg_nor)
            mean_abn = torch.mean(deg_abn)

            mean_nors_nor.append(mean_nor)
            mean_nors_abn.append(mean_abn)

        loss_mean_abn = sum(mean_nors_abn) / (b // 2)
        loss_mean_nor = sum(mean_nors_nor) / (b // 2)

        loss_mean = torch.norm(loss_mean_nor - loss_mean_abn)

        # pdb.set_trace()

        return loss_mean
