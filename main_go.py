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
import itertools


if __name__ == '__main__':

    args = option.parser.parse_args()
    seed_everything(args.seed)

    args.wandb = False

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



    args = option.parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    paren_dir = "logs"
    dir_path = f"{args.dataset}"
    log_path = os.path.join(paren_dir, dir_path)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(f"{args.dataset}-log-{args.ablation}-{args.modal}.txt", mode='w'),
                                  logging.StreamHandler()])

    class CombinedLoss(nn.Module):
        def __init__(self, w_var=1., w_normal=1., threshold=0.5, k=10):
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


    m_values = np.arange(1, 10)
    n_values = np.arange(1, 10)
    combinations = list(itertools.product(m_values, n_values))
    print(f"Total combinations to test: {len(combinations)}")

    for i in range(len(combinations)):
        num_combinations = len(combinations)
        m, n = combinations[i]
        plot_label = str(f"{args.dataset}: m={m}, n={n}")
        if args.wandb:
            wandb.init(
                project="GraphFusion",
                name=f"{plot_label}_variance_ablation",
                reinit=True
            )
        model = graphfusion.GraphClassifiction(batch_size=32, m=m, n=n)
        criterion = CombinedLoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        logging_message_test = f"go setup: m={m}, n={n}, w_var=1, threshold=0.5, k=10, graph operator: {args.go}"
        logging.info(logging_message_test)

        train(train_nloader, train_aloader, test_loader, model, optimizer, criterion, device, args, scheduler, m,
              n, save_dir=f'./ckpt', num_epochs=args.epoch)

        if args.wandb:
            wandb.finish()



