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
from con_loss import Connectivity_loss
import pdb
import itertools
import wandb



if __name__ == '__main__':
    # parse arguments
    args = option.parser.parse_args()
    # config = Config(args)
    # wandb setup
    args.wandb = False

    # set random seed
    # torch.manual_seed(4869)
    seed_everything(args.seed)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # load data
    # TODO: havent check dataloader
    print('LOADING NORMAL')
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=False,
                               generator=torch.Generator(device=device))

    print('LOADING ABNORMAL')
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=False,
                               generator=torch.Generator(device=device))

    test_loader = DataLoader(Dataset(args, test_mode=True), batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=False)

    # TODO: can be separated to loss.py
    class CombinedLoss(nn.Module):
        def __init__(self, w_normal=1., w_var=1., w_mean=0., threshold=0.5, k=10):
            super().__init__()
            self.k = k
            self.threshold = threshold
            self.w_normal = w_normal
            self.w_var = w_var
            self.w_mean = w_mean
            self.normalLoss = NormalLoss()
            self.varLoss = VarianceLoss(k=self.k, threshold=self.threshold)
            # self.meanLoss = MeanLoss(k=self.k, threshold=self.threshold)


        def forward(self, result, label):
            loss = {}

            all_scores = result['scores']

            # normal_loss = self.normalLoss(pre_normal_scores)

            normal_loss = self.normalLoss(all_scores, label)

            loss['normal_loss'] = normal_loss

            var_loss = self.varLoss(result)
            # mean_loss = self.meanLoss(result)


            # loss['total_loss'] = self.w_normal * normal_loss + self.w_var * var_loss + self.w_mean * mean_loss
            loss['total_loss'] = self.w_normal * normal_loss + self.w_var * var_loss

            # pdb.set_trace()

            return loss['total_loss'], loss


    #TODO: can be add to parser for parallel training
    m_values = np.arange(4, 8)
    n_values = np.arange(4, 8)
    combinations = list(itertools.product(m_values, n_values))

    for (m, n) in combinations:
        plot_label = str(f"{args.dataset}: m={m}, n={n}")
        if args.wandb:
            wandb.init(
                project = "GraphFusion",
                name    = f"{plot_label}_plot",
                tag     = f"qx_dev",
                reinit  = True,
            )
        model = graphfusion.GraphClassifiction(m=m, n=n)
        criterion = CombinedLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        train(train_nloader, train_aloader, test_loader, model, optimizer, criterion, device, args, scheduler, m, n,
               save_dir='./ckpt')
        if args.wandb:
            wandb.finish()


    # model = graphfusion.GraphClassifiction(batch_size=32, m=4, n=4)
    #
    # criterion = CombinedLoss()
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=0.001, weight_decay=1e-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    #
    # train(train_nloader, train_aloader, test_loader, model, optimizer, criterion, device, viz, args, scheduler, save_dir='./ckpt', num_epochs=1000)