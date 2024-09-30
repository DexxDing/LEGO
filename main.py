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
            self.meanLoss = MeanLoss(k=self.k, threshold=self.threshold)


        def forward(self, result, label):
            loss = {}

            all_scores = result['scores']

            # normal_loss = self.normalLoss(pre_normal_scores)

            normal_loss = self.normalLoss(all_scores, label)

            loss['normal_loss'] = normal_loss

            var_loss = self.varLoss(result)
            mean_loss = self.meanLoss(result)


            loss['total_loss'] = self.w_normal * normal_loss + self.w_var * var_loss + self.w_mean * mean_loss

            # pdb.set_trace()

            return loss['total_loss'], loss

    # ablations

    m_values = np.arange(4, 5)
    n_values = np.arange(4, 5)
    combinations = list(itertools.product(m_values, n_values))
    print(f"Total combinations to test: {len(combinations)}")


    for i in range(len(combinations)):
        num_combinations = len(combinations)
        m, n = combinations[i]
        plot_label = str(f"{args.dataset}: m={m}, n={n}")
        wandb.init(
            project="GraphFusion",
            name=f"{plot_label}",
            reinit=True
        )
        model = graphfusion.GraphClassifiction(batch_size=32, m=m, n=n)
        criterion = CombinedLoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=0.001, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        train(train_nloader, train_aloader, test_loader, model, optimizer, criterion, device, viz, args, scheduler, m, n,
               save_dir='./ckpt', num_epochs=10)

        wandb.finish()


    # # testing variance loss coefficient
    # w_variance_loss = [0.001, 0.1, 0.5, 1, 5, 10]
    # for i in range(len(w_variance_loss)):
    #     w_var = w_variance_loss[i]
    #     m = 4
    #     n = 4
    #     plot_label = str(f"{args.dataset}: w_var:{w_var}")
    #     wandb.init(
    #         project="GraphFusion",
    #         name=f"{plot_label}_new_var",
    #         reinit=True
    #     )
    #
    #     model = graphfusion.GraphClassifiction(batch_size=32, m=m, n=n)
    #     criterion = CombinedLoss(w_var=w_var)
    #     optimizer = optim.Adam(model.parameters(),
    #                            lr=0.001, weight_decay=1e-3)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    #     train(train_nloader, train_aloader, test_loader, model, optimizer, criterion, device, viz, args, scheduler,
    #           m, n,
    #           save_dir='./ckpt', num_epochs=200)
    #
    #     wandb.finish()







    # model = graphfusion.GraphClassifiction(batch_size=32, m=4, n=4)
    #
    # criterion = CombinedLoss()
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=0.001, weight_decay=1e-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    #
    # train(train_nloader, train_aloader, test_loader, model, optimizer, criterion, device, viz, args, scheduler, save_dir='./ckpt', num_epochs=1000)


