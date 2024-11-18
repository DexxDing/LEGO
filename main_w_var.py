import os

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
import logging
import torch.nn as nn
from normal_loss import NormalLoss
from variance_loss import VarianceLoss
from con_loss import Connectivity_loss
import matplotlib.pyplot as plt
import wandb

if __name__ == '__main__':

    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)

    args.wandb = False
    args.w_var = True

    # set random seed
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print('LOADING NORMAL')
    train_nloader = DataLoader(
        Dataset(args, test_mode=False, is_normal=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False,
        generator=torch.Generator(device=device)
    )

    print('LOADING ABNORMAL')
    train_aloader = DataLoader(
        Dataset(args, test_mode=False, is_normal=False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False,
        generator=torch.Generator(device=device)
    )

    test_loader = DataLoader(
        Dataset(args, test_mode=True), batch_size=1, shuffle=False, num_workers=0,
        pin_memory=False
    )


    os.makedirs('ckpt', exist_ok=True)
    args = option.parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    paren_dir = "logs"
    dir_path = f"{args.dataset}"
    log_path = os.path.join(paren_dir, dir_path)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(f"{args.dataset}-log-{args.ablation}.txt", mode='w'),
                                  logging.StreamHandler()])

    ckpt_parent = "ckpt"
    dataset_ckpt_path = os.path.join(ckpt_parent, f"{args.dataset}_{args.ablation}ckpt")
    os.makedirs(dataset_ckpt_path, exist_ok=True)

    class CombinedLoss(nn.Module):
        def __init__(self, w_var, w_normal=1., threshold=0.5, k=10):
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

            normal_loss = self.normalLoss(all_scores, label)
            loss['normal_loss'] = normal_loss

            var_loss = self.varLoss(result)
            loss['total_loss'] = self.w_normal * normal_loss + self.w_var * var_loss

            return loss['total_loss'], loss

    w_variance_loss = [0, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0]
    for i in range(len(w_variance_loss)):
        m = 4
        n = 4
        w_var = w_variance_loss[i]
        plot_label = f"{args.dataset}: w_var:{w_var}"
        if args.wandb:
            wandb.init(
                project="GraphFusion",
                name=f"{plot_label}_variance_ablation",
                reinit=True
            )

        model = graphfusion.GraphClassifiction(batch_size=args.batch_size, m=m, n=n).to(device)
        criterion = CombinedLoss(w_var=w_var).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        logging_message_test = f"w_var setup: m={m}, n={n}, w_var={w_var}, threshold=0.5, k=10"
        logging.info(logging_message_test)

        train(
            train_nloader, train_aloader, test_loader, model, optimizer,
            criterion, device, args, scheduler, m, n,
            save_dir=f'./ckpt/{args.dataset}_ckpt', num_epochs=args.epoch
        )

        if args.wandb:
            wandb.finish()
