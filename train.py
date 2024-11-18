import pdb

import numpy as np
import torch.nn as nn
import torch
import os
import logging
from sklearn.metrics import accuracy_score, f1_score
from webencodings import labels

import option
from test_10crop import test
import torch.nn.functional as F
import itertools
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(4869)





def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN values found in {name}")
        os._exit(1)
    else:
        print(f"No NaN values found in {name}")


def check_for_inf(tensor, name):
    if torch.isinf(tensor).any():
        print(f"Inf values found in {name}")
        os._exit(1)
    else:
        print(f"No Inf values found in {name}")


args = option.parser.parse_args()





def train(train_nloader, train_aloader, test_loader, model, optimizer, criterion, device, args, scheduler, m , n, save_dir=None, num_epochs=10):
    with ((torch.set_grad_enabled(True))):


        for epoch in range(num_epochs):
            model.train()
            total_batches = 0
            total_loss = 0.0
            loss_sum = 0
            acc_sum = 0


            # iterators for both loaders
            nloader_iter = iter(train_nloader)
            aloader_iter = iter(train_aloader)

            batch_idx = 0

            # loop until both iterators are exhausted
            while True:
                try:
                    batch_n = next(nloader_iter)
                except StopIteration:
                    break

                try:
                    batch_a = next(aloader_iter)
                except StopIteration:
                    break

                batch_idx += 1
                print(f'Batch {batch_idx}')

                v_features_n, t_features_n, labels_n = [n.to(device) for n in batch_n]
                v_features_a, t_features_a, labels_a = [a.to(device) for a in batch_a]


                cut = min(v_features_n.shape[0], v_features_a.shape[0])
                v_features_n, t_features_n, labels_n = v_features_n[:cut], t_features_n[:cut], labels_n[:cut]
                v_features_a, t_features_a, labels_a = v_features_a[:cut], t_features_a[:cut], labels_a[:cut]


                cut_snippet = min(v_features_n.shape[2], v_features_a.shape[2])
                v_features_n, t_features_n = v_features_n[:, :, :cut_snippet, :], \
                                            t_features_n[:, :, :cut_snippet, :]
                v_features_a, t_features_a = v_features_a[:, :, :cut_snippet, :], \
                                            t_features_a[:, :, :cut_snippet, :]

                v_features = torch.cat((v_features_n, v_features_a), dim=0)
                t_features = torch.cat((t_features_n, t_features_a), dim=0)

                labels = torch.cat((labels_n, labels_a), dim=0)



                v_features.requires_grad = False
                t_features.requires_grad = False


                print('Batch dimension: ', v_features.shape, t_features.shape)


                output = model(v_features, t_features)
                scores = torch.max(output['scores'], dim=-1)[0]
                scores = scores.detach().cpu().numpy()
                fusion_matrix = output['fusion_matrix']
                a_fused = output['bn_results']['fused_graphs']
                av = output['bn_results']['av']
                at = output['bn_results']['at']

                # a_fused = a_fused.detach().cpu().numpy()
                # graph_path = f"./graphs/train/graph_{args.dataset}-{batch_idx}1-train.npy"
                # np.save(graph_path, a_fused)
                #
                # av = av.detach().cpu().numpy()
                # av_path = f"./graphs/train/av/av_{args.dataset}_{batch_idx}1-train.npy"
                # np.save(av_path, av)
                #
                # at = at.detach().cpu().numpy()
                # at_path = f"./graphs/train/at/at_{args.dataset}_{batch_idx}1-train.npy"
                # np.save(at_path, at)



                loss, cost = criterion(output, labels)


                total_loss += loss.item()
                total_batches += 1

                log_message = f'Epoch {epoch + 1} Batch {batch_idx} Loss: {loss:.4f} '
                print(log_message)
                logging.info(log_message)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted_labels = (scores >= 0.5).astype(int)

                labels_acc = labels.detach().cpu().numpy()


                acc = accuracy_score(y_true=labels_acc, y_pred=predicted_labels)

                acc_sum += acc

            fusion_matrix = fusion_matrix.detach().cpu().numpy()
            fusion_matrix_path = f"./fusion_matrix/fusion_matrix_{epoch+1}_{args.dataset}_m{m}_n{n}.npy"
            np.save(fusion_matrix_path, fusion_matrix)




            # scheduler.step()

            avg_acc = acc_sum / total_batches
            average_loss = total_loss / total_batches
            # viz.plot_lines('loss', average_loss, flag, color=color, label=plot_label)
            # viz.plot_lines('train acc', avg_acc)

            if args.wandb:
                wandb.log({
                    "Loss": average_loss,
                }, step=epoch + 1)



            model_path = os.path.join(save_dir, f'{args.dataset}_model_epoch_{epoch + 1}_{average_loss:.4f}.pth')
            model_state = model.state_dict()
            torch.save(model_state, model_path)
            graph_path = f'./graphs/graph{args.dataset}_epoch{epoch}_'
            torch.save(a_fused, graph_path)

            if (epoch + 1) % 1 == 0:
                model.eval()
                print(f"Epoch {epoch + 1}: Conducting testing...")
                with torch.no_grad():
                    model.load_state_dict(model_state)
                    model.to(device)
                    test_auc, test_ap = test(test_loader, model, args, device, epoch)
                print(f"Epoch {epoch + 1} Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
                if args.wandb:
                    wandb.log({
                        "Test AUC": test_auc
                    }, step=epoch + 1)

                if args.wandb:
                    wandb.log({
                        "Test AP": test_ap
                    }, step=epoch + 1)

                logging_message_test = f"Task {args.modal} Epoch {epoch + 1} Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}"
                logging.info(logging_message_test)
