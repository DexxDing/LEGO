import pdb

import torch
from torch import nn


class Connectivity_loss(nn.Module):
    def __init__(self, sigma, k=10):
        super(Connectivity_loss, self).__init__()
        self.sigma = sigma
        self.k = k

    def forward(self, results):
        features = results['bn_results']['fused_graphs']

        b, c, t = features.size()

        g_nor = features[0:b // 2]
        g_abn = features[b // 2:]


        g_abn_flatten = g_abn.view(b//2, -1)
        topk_abn = torch.topk(g_abn_flatten, k=self.k, dim=-1)[0]




        con_nors_nor = []
        con_nors_abn = []
        for i in range(b // 2):
            con_nor = torch.mean(g_nor[i, :, :])
            # con_nor = torch.mean(con_nor, dim=0)
            con_nors_nor.append(con_nor)


            con_abn = torch.mean(topk_abn[i, :])
            # con_abn = torch.mean(con_abn, dim=0)
            con_nors_abn.append(con_abn)

        loss_con_abn = sum(con_nors_abn) / (b // 2)
        loss_con_nor = sum(con_nors_nor) / (b // 2)

        # pdb.set_trace()


        # loss = loss_con_abn - loss_con_nor
        loss = (loss_con_abn - (loss_con_nor + self.sigma))
        # loss = (1 - loss_con_nor)

        # pdb.set_trace()


        return loss
