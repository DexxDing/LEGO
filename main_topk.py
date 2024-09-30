import numpy as np
import torch.optim as optim
from train import train
from utils import *
from dataset import Dataset
from utils import seed_everything
import option
from config import Config
from torch.utils.data import DataLoader
import torch
import graphfusion
import torch.nn as nn
from normal_loss import NormalLoss
from variance_loss import VarianceLoss
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

    class CombinedLossTopK(nn.Module):
        def __init__(self, k, threshold=0.5, w_normal=1., w_var=1.):
            super().__init__()
            self.k = k
            self.threshold = threshold
            self.w_normal = w_normal
            self.w_var = w_var
            self.normalLoss = NormalLoss()
            self.varLoss = VarianceLoss(k=self.k, threshold=self.threshold)

        def forward(self, result, label):
            loss = {}

            all_scores = result['scores']

            # normal_loss = self.normalLoss(pre_normal_scores)

            normal_loss = self.normalLoss(all_scores, label)

            loss['normal_loss'] = normal_loss

            var_loss = self.varLoss(result)

            loss['total_loss'] = self.w_normal * normal_loss + self.w_var * var_loss

            # pdb.set_trace()

            return loss['total_loss'], loss

    # testing mix loss coefficient
    ks = np.linspace(1, 20, 25).tolist()
    ks = ks.astype(int)
    for i in range(len(ks)):
        k = ks[i]
        m = 4
        n = 4
        plot_label = str(f"{args.dataset}: k:{k}")
        wandb.init(
            project="GraphFusion",
            name=f"{plot_label}_topk-ablation",
            reinit=True
        )
        model = graphfusion.GraphClassifiction(batch_size=32, m=m, n=n)
        criterion = CombinedLossTopK(k=k)
        optimizer = optim.Adam(model.parameters(),
                               lr=0.001, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        train(train_nloader, train_aloader, test_loader, model, optimizer, criterion, device, viz, args, scheduler,
              m, n, save_dir='./ckpt', num_epochs=10)

        wandb.finish()





