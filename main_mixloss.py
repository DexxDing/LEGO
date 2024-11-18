import numpy as np
import torch.optim as optim
from train import train
from utils import *
from dataset import Dataset
from utils import seed_everything
import option
from config import Config
from torch.utils.data import  DataLoader
import torch
import graphfusion
import torch.nn as nn
from normal_loss import NormalLoss
from variance_loss import VarianceLoss
from mean_loss import MeanLoss
from con_loss import Connectivity_loss
import pdb
import itertools
import matplotlib.pyplot as plt
import matplotlib
import wandb

if __name__ == '__main__':
    torch.manual_seed(4869)
    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('LOADING NORMAL')
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=False,
                               generator=torch.Generator(device='cuda'))

    print('LOADING ABNORMAL')
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=False,
                               generator=torch.Generator(device='cuda'))

    test_loader = DataLoader(Dataset(args, test_mode=True), batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=False)


    viz_name = 'BNGF'
    viz = Visualizer(env=viz_name, use_incoming_socket=False)

    class CombinedLossMix(nn.Module):
        def __init__(self, w_normal=1., w_var=1., threshold=0.5, k=10):
            super().__init__()
            self.k = k
            self.threshold = threshold
            self.w_normal = w_normal
            self.w_var = w_var
            self.normalLoss = NormalLoss()
            self.meanLoss = MeanLoss(k=self.k, threshold=self.threshold)
            self.varLoss = VarianceLoss(k=self.k, threshold=self.threshold)

        def forward(self, result, label):
            loss = {}

            all_scores = result['scores']

            # normal_loss = self.normalLoss(pre_normal_scores)

            normal_loss = self.normalLoss(all_scores, label)

            loss['normal_loss'] = normal_loss

            var_loss = self.varLoss(result)
            mean_loss = self.meanLoss(result)


            loss['total_loss'] = self.w_normal * normal_loss + self.w_var * var_loss + (1-self.w_var) * mean_loss

            # pdb.set_trace()

            return loss['total_loss'], loss


    # testing mix loss coefficient
    w_variance_loss = np.linspace(0.001, 10, 25).tolist()
    for i in range(len(w_variance_loss)):
        w_var = w_variance_loss[i]
        w_mean = 1.0 - w_var
        m = 4
        n = 4
        plot_label = str(f"{args.dataset}: w_var:{w_var}, w_nor:{w_mean}")
        wandb.init(
            project="GraphFusion",
            name=f"{plot_label}_mixloss-ablation",
            reinit=True
        )
        model = graphfusion.GraphClassifiction(batch_size=32, m=m, n=n)
        criterion = CombinedLossMix(w_var=w_var)
        optimizer = optim.Adam(model.parameters(),
                               lr=0.001, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        train(train_nloader, train_aloader, test_loader, model, optimizer, criterion, device, viz, args, scheduler, m, n, save_dir='./ckpt', num_epochs=10)

        wandb.finish()