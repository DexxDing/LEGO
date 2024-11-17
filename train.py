import pdb

import numpy as np
import torch.nn as nn
import torch
import os
import logging
from sklearn.metrics import accuracy_score, f1_score
from webencodings import labels

from test_10crop import test
import torch.nn.functional as F
import itertools
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import random

torch.manual_seed(4869)

logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler("ucf_training_log.txt", mode='w'), logging.StreamHandler()])


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

                v_features_n, v_features_n2, v_features_n3, v_features_n4, t_features_n, labels_n = [n.to(device) for n in batch_n]
                v_features_a, v_features_a2, v_features_a3, v_features_a4, t_features_a, labels_a = [a.to(device) for a in batch_a]


                cut = min(v_features_n.shape[0], v_features_a.shape[0])
                v_features_n, v_features_n2, v_features_n3, v_features_n4, t_features_n, labels_n = v_features_n[:cut], v_features_n2[:cut], v_features_n3[:cut], v_features_n4[:cut], \
                    t_features_n[:cut], labels_n[:cut]
                v_features_a, v_features_a2, v_features_a3, v_features_a4, t_features_a, labels_a = v_features_a[:cut], v_features_a2[:cut], v_features_a3[:cut], v_features_a4[:cut], \
                    t_features_a[:cut], labels_a[:cut]


                cut_snippet = min(v_features_n.shape[2], v_features_a.shape[2])
                v_features_n, v_features_n2, v_features_n3, v_features_n4, t_features_n = v_features_n[:, :, :cut_snippet, :], \
                                                                            v_features_n2[:, :, :cut_snippet, :],\
                                                                            v_features_n3[:, :, :cut_snippet, :],\
                                                                            v_features_n4[:, :, :cut_snippet, :],\
                                                                            t_features_n[:, :, :cut_snippet, :]
                v_features_a, v_features_a2, v_features_a3, v_features_a4, t_features_a = v_features_a[:, :, :cut_snippet, :], \
                                                                            v_features_a2[:, :, :cut_snippet, :],\
                                                                            v_features_a3[:, :, :cut_snippet, :],\
                                                                            v_features_a4[:, :, :cut_snippet, :],\
                                                                            t_features_a[:, :, :cut_snippet, :]

                v_features = torch.cat((v_features_n, v_features_a), dim=0)
                # v_features2 = torch.cat((v_features_n2, v_features_a2), dim=0)
                # v_features3 = torch.cat((v_features_n3, v_features_a3), dim=0)
                v_features4 = torch.cat((v_features_n4, v_features_a4), dim=0)
                t_features = torch.cat((t_features_n, t_features_a), dim=0)

                labels = torch.cat((labels_n, labels_a), dim=0)


                dim0, dim1, dim2, dim3 = v_features4.shape
                # create a noise feature here
                v_features4 = torch.rand(dim0, dim1, dim2, dim3)




                # feature_list = [v_features, v_features2, v_features3, t_features]
                feature_list = [v_features, v_features4, t_features]


                selected_indices = random.sample(range(len(feature_list)), 2)

                selected_features = [feature_list[i] for i in selected_indices]

                indices = list(selected_indices)









                v_features.requires_grad = False
                t_features.requires_grad = False


                print('Batch dimension: ', selected_features[0].shape, selected_features[1].shape)

                # pdb.set_trace()


                output = model(selected_features, indices=indices)
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

            wandb.log({
                "Loss": average_loss,
            }, step=epoch + 1)

            # fig, ax = plt.subplots(figsize=(8, 6))
            # sns.heatmap(
            #     fusion_matrix,
            #     annot=True,
            #     fmt=".2f",
            #     cmap='coolwarm',  # Diverging color map suitable for negative and positive values
            #     center=0,  # Center the colormap at zero
            #     ax=ax,
            #     cbar=True,
            #     linewidths=.5,  # Add lines between cells for clarity
            #     linecolor='gray'
            # )
            # ax.set_title(f'Fusion Matrix - Epoch {epoch + 1}')
            # ax.set_xlabel('Matrix Column')
            # ax.set_ylabel('Matrix Row')

            # Log the heatmap image to wandb
            # wandb.log({"Fusion Matrix": wandb.Image(fig)})

            # plt.close(fig)



            model_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}_{average_loss:.4f}.pth')
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
                # viz.plot_lines('test auc', test_auc)
                wandb.log({
                    "Test AUC": test_auc
                }, step=epoch + 1)

                # # viz.plot_lines('test ap', test_ap, color=color, label=plot_label)
                # wandb.log({
                #     "Test AP": test_ap
                # }, step=epoch + 1)

                logging_message_test = f"Epoch {epoch + 1} Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}"
                logging.info(logging_message_test)
