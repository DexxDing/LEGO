import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceLoss(nn.Module):
    def __init__(self):
        super(VarianceLoss, self).__init__()


    def forward(self, results):
        features = results['bn_results']['fused_graphs']

        b, c, t = features.size()

        g_nor = features[0:b // 2]
        g_abn = features[b // 2:]


        var_nors_nor = []
        var_nors_abn = []
        for i in range(b // 2):
            var_nor = torch.var(g_nor[i, :, :])
            # var_nor = torch.mean(var_nor, dim=0)
            var_nors_nor.append(var_nor)

            var_abn = torch.var(g_abn[i, :, :])
            # var_abn = torch.mean(var_abn, dim=0)
            var_nors_abn.append(var_abn)

        loss_var_abn = sum(var_nors_abn) / (b // 2)
        loss_var_nor = sum(var_nors_nor) / (b // 2)


        # pdb.set_trace()

        loss_var = loss_var_nor / (loss_var_abn + 1e-5)

        return loss_var

