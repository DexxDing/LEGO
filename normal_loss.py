import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()
        self.k = 3
        
    def forward(self, scores, labels):
        '''
        normal_scores: [bs, pre_k]
        '''


        # loss_normal = torch.norm(normal_scores, dim=1, p=2)
        scores = torch.max(scores, dim=-1)[0]
        # scores = torch.topk(scores, self.k, dim=-1)[0]

        loss_normal = self.loss(scores, labels)




        return loss_normal.mean()