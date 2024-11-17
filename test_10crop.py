import pdb

from utils import *
from config import *
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc, \
    confusion_matrix

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


def get_metrics(frame_predict, frame_gt):
    metrics = {}

    fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
    metrics['AUC'] = auc(fpr, tpr)

    precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
    metrics['AP'] = auc(recall, precision)

    return metrics


def get_predicts(feat_list, net):
    frame_predict = []


    # fv = fv.cuda()
    # ft = ft.cuda()
    res, av, at, a_fused = net(feat_list)

    a_predict = res.cpu().numpy().mean(0)

    # fpre_ = np.repeat(a_predict, 16)
    frame_predict.append(a_predict)

    frame_predict = np.concatenate(frame_predict, axis=0)
    return frame_predict, av, at, a_fused


def test(loader, model, args, device, epoch):
    test_gt = get_gt(args.dataset, args.gt)
    print('Original gt:', len(test_gt))
    all_pred = []
    all_gt = []

    window_size = 32
    gt_index = 0


    with torch.no_grad():
        model.eval()
        for batch_index, batch_features in enumerate(loader):
            # if batch_index < 3 and batch_index > 0:
            print(f'Batch {batch_index + 1}')
            v_features, v_features2, v_features3, v_features4, t_features = [tensor.to(device) for tensor in batch_features]


            v_features = v_features.permute(0, 2, 1, 3)
            t_features = t_features.permute(0, 2, 1, 3)
            v_features2 = v_features2.permute(0, 2, 1, 3)
            v_features3 = v_features3.permute(0, 2, 1, 3)
            v_features4 = v_features4.permute(0, 2, 1, 3)


            original_batch, crops, num_segments, feature_dim = v_features.shape
            feature_dim_t = t_features.shape[3]

            # pdb.set_trace()
            cut = min(v_features.shape[2], t_features.shape[2])
            v_features = v_features[:, :, :cut, :]
            v_features2 = v_features2[:, :, :cut, :]
            v_features3 = v_features3[:, :, :cut, :]
            v_features4 = v_features4[:, :, :cut, :]
            t_features = t_features[:, :, :cut, :]

            # pdb.set_trace()

            num_segments = v_features.shape[2]

            v_features = v_features.view(-1, num_segments, feature_dim)
            v_features2 = v_features2.view(-1, num_segments, v_features2.shape[3])
            v_features3 = v_features3.view(-1, num_segments, v_features3.shape[3])
            v_features4 = v_features4.view(-1, num_segments, v_features4.shape[3])
            t_features = t_features.view(-1, num_segments, feature_dim_t)

            # add noise
            dim0, dim1, dim2 = v_features4.shape
            # create a noise feature here
            v_features4 = torch.rand(dim0, dim1, dim2)


            print(f'feature size before padding, {v_features.shape}, {t_features.shape}')

            # sample out the gt for the given video
            video_gt = test_gt[gt_index:gt_index + num_segments]
            # keep track of the gt index
            gt_index += num_segments

            if num_segments < window_size:
                # padding for segments < 32
                print('repeat padding')
                repeat_times = window_size // num_segments + 1
                # repeat the segments from the start until it reaches 32


                # pdb.set_trace()



                v_features = v_features.repeat(1, repeat_times, 1)[:, :window_size, :]
                v_features2 = v_features2.repeat(1, repeat_times, 1)[:, :window_size, :]
                v_features3 = v_features3.repeat(1, repeat_times, 1)[:, :window_size, :]
                v_features4 = v_features4.repeat(1, repeat_times, 1)[:, :window_size, :]
                t_features = t_features.repeat(1, repeat_times, 1)[:, :window_size, :]

                print(f'feature size, {v_features.shape}, {t_features.shape}')

                v_features = v_features.view(1, crops, v_features.shape[1], v_features.shape[2])
                v_features2 = v_features2.view(1, crops, v_features2.shape[1], v_features2.shape[2])
                v_features3 = v_features3.view(1, crops, v_features3.shape[1], v_features3.shape[2])
                v_features4 = v_features4.view(1, crops, v_features4.shape[1], v_features4.shape[2])
                t_features = t_features.view(1, crops, t_features.shape[1], t_features.shape[2])



                # f_v = v_features.squeeze(0)
                # f_v = F.normalize(f_v, p=2, dim=2)
                # f_t = t_features.squeeze(0)
                # f_t = F.normalize(f_t, p=2, dim=2)
                #
                # a_v = (torch.bmm(f_v, f_v.transpose(1, 2)))
                # a_t = (torch.bmm(f_t, f_t.transpose(1, 2)))

                # av = a_v.detach().cpu().numpy()
                # av_path = f"./graphs/train/av/av_{args.dataset}_{batch_index}1.npy"
                # np.save(av_path, av)
                #
                # at = a_t.detach().cpu().numpy()
                # at_path = f"./graphs/train/at/at_{args.dataset}_{batch_index}1.npy"
                # np.save(at_path, at)


                # pdb.set_trace()

                # feature_list = [v_features, v_features2, v_features3, t_features]
                feature_list = [v_features, v_features4, t_features]



                # get snippet level results
                snippet_scores, av, at, a_fused = get_predicts(feature_list, model)
                # check_for_nan(snippet_scores, 'snippet_scores')
                # snippet_scores = torch.squeeze(snippet_scores, 1)
                print('snippet scores dimension', snippet_scores.shape)

                # a_fused = a_fused.detach().cpu().numpy()
                # graph_path = f"./graphs/graph_{args.dataset}-{epoch}-batch{batch_index}1.npy"
                # np.save(graph_path, a_fused)
                #
                # av = av.detach().cpu().numpy()
                # av_path = f"./graphs/av/av_{args.dataset}_{epoch}-batch{batch_index}1.npy"
                # np.save(av_path, av)
                #
                # at = at.detach().cpu().numpy()
                # at_path = f"./graphs/at/at_{args.dataset}_{epoch}-batch{batch_index}1.npy"
                # np.save(at_path, at)



                # new gt based on the paddings
                all_pred.extend(snippet_scores[:num_segments])
                all_gt.extend(video_gt)
            else:
                # sliding windows for segments > 32
                for i in range(0, num_segments, 32):
                    print(f'windows: {i}')
                    end_idx = i + window_size

                    if end_idx > num_segments:
                        # shift back padding
                        start_idx = num_segments - window_size
                        v_window = v_features[:, start_idx:num_segments, :]
                        v_window2 = v_features2[:, start_idx:num_segments, :]
                        v_window3 = v_features3[:, start_idx:num_segments, :]
                        v_window4 = v_features4[:, start_idx:num_segments, :]
                        t_window = t_features[:, start_idx:num_segments, :]
                        print(f'v_window.shape: {v_window.shape}, t_window.shape: {t_window.shape}')




                    else:
                        # sliding window for segments > 32 and can be divided by 32
                        v_window = v_features[:, i:end_idx, :]
                        v_window2 = v_features2[:, i:end_idx, :]
                        v_window3 = v_features3[:, i:end_idx, :]
                        v_window4 = v_features4[:, i:end_idx, :]
                        t_window = t_features[:, i:end_idx, :]
                        print(f'v_window.shape: {v_window.shape}, t_window.shape: {t_window.shape}')


                    v_window = v_window.view(1, crops, v_window.shape[1], v_window.shape[2])
                    v_window2 = v_window2.view(1, crops, v_window2.shape[1], v_window2.shape[2])
                    v_window3 = v_window3.view(1, crops, v_window3.shape[1], v_window3.shape[2])
                    v_window4 = v_window4.view(1, crops, v_window4.shape[1], v_window4.shape[2])
                    t_window = t_window.view(1, crops, t_window.shape[1], t_window.shape[2])

                    # pdb.set_trace()

                    # feature_list = [v_window, v_window2, v_window3, t_window]
                    feature_list = [v_window, v_window4, t_window]


                    snippet_scores, av, at, a_fused = get_predicts(feature_list, model)
                    # check_for_nan(snippet_scores, 'snippet_scores')
                    # snippet_scores = torch.squeeze(snippet_scores, 1)

                    all_pred.extend(snippet_scores)

                    if end_idx > num_segments:
                        print(f'end>seg{end_idx > num_segments}')
                        all_gt.extend(video_gt[start_idx:num_segments])
                    else:
                        all_gt.extend(video_gt[i:end_idx])
                    # pdb.set_trace()

                    if end_idx >= num_segments:
                        break

                a_fused = a_fused.detach().cpu().numpy()
                # graph_path = f"./graphs/graph_{args.dataset}-{epoch}-batch{batch_index}.npy"
                # np.save(graph_path, a_fused)

                av = av.detach().cpu().numpy()
                # av_path = f"./graphs/av/av_{args.dataset}_{epoch}-batch{batch_index}.npy"
                # np.save(av_path, av)

                at = at.detach().cpu().numpy()
                # at_path = f"./graphs/at/at_{args.dataset}_{epoch}-batch{batch_index}.npy"
                # np.save(at_path, at)





        all_pred = np.array(all_pred)
        all_gt = np.array(all_gt)

        all_pred_plot = all_pred[:100]
        all_gt_plot = all_gt[:100]

        # all_pred_plot = all_pred_plot / all_pred_plot.max()
        #
        #
        # plot_prediction_vs_gt(all_gt_plot, all_pred_plot, save_path=f'./preds/pred{epoch}.png')



        print('all pred', all_pred.shape)
        print('all gt', all_gt.shape)

        np.save('all_pred.npy', all_pred)
        np.save('all_gt.npy', all_gt)



        auc = roc_auc_score(y_true=all_gt, y_score=all_pred)
        ap = average_precision_score(y_true=all_gt, y_score=all_pred)
        fpr, tpr, thresholds = roc_curve(y_true=all_gt, y_score=all_pred)

        print('ROC AUC:', auc)
        print('Average Precision:', ap)


        return auc, ap