import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceLoss(nn.Module):
    def __init__(self, k, threshold):
        super(VarianceLoss, self).__init__()
        self.k = k
        self.threshold = threshold


    def forward(self, results):
        features = results['bn_results']['fused_graphs']

        b, c, t = features.size()

        g_nor = features[0:b // 2]
        g_abn = features[b // 2:]

        var_nors_nor = []
        var_nors_abn = []
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

            var_nor = torch.var(deg_nor)
            var_abn = torch.var(deg_abn)

            var_nors_nor.append(var_nor)
            var_nors_abn.append(var_abn)


        loss_var_abn = sum(var_nors_abn) / (b // 2)
        loss_var_nor = sum(var_nors_nor) / (b // 2)



        loss_var = loss_var_nor - loss_var_abn

        # pdb.set_trace()


        return loss_var

