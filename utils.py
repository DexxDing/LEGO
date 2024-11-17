import pdb

import visdom
import numpy as np
import torch
import random, os
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt


def plot_prediction_vs_gt(gt, pred, save_path=None):
    """
    Plot prediction vs ground truth data.

    Args:
    gt (np.array): Ground truth data of shape (100,)
    pred (np.array): Prediction data of shape (100,)
    save_path (str, optional): Path to save the plot. If None, the plot is displayed instead.
    """
    # Ensure the inputs are numpy arrays
    gt = np.array(gt)
    pred = np.array(pred)

    # Create x-axis (assuming uniform spacing)
    x = np.arange(len(gt))

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot ground truth
    plt.plot(x, gt, color='blue', label='Ground Truth')

    # Plot prediction
    plt.plot(x, pred, color='red', label='Prediction')

    # Fill the area between prediction and ground truth
    # plt.fill_between(x, gt, pred, color='red', alpha=0.3)

    # Set labels and title
    plt.xlabel('Snippets')
    plt.ylabel('Anomaly Scores')
    plt.title('Prediction vs Ground Truth')
    plt.legend()

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Tight layout
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''


        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))

    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)

    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int32)  # len=33,存入要取的frame index
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)  # r[i]:r[i+1]这些feat求平均
        else:
            new_feat[i, :] = feat[r[i], :]  # 不足32帧补全
    return new_feat

# def process_feat(feat, length, visualize=False):
#     new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
#     r = np.linspace(0, len(feat), length + 1, dtype=np.int32)
#     for i in range(length):
#         if r[i] != r[i + 1]:
#             new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
#         else:
#             new_feat[i, :] = feat[r[i], :]
#
#     if visualize:
#         plt.figure(figsize=(12, 6))
#
#
#         plt.plot(np.arange(len(feat)), feat.mean(-1), 'bo-')
#
#         plt.plot(np.linspace(0, len(feat), length), new_feat.mean(axis=-1), 'ro-')
#
#         plt.title(f'Feature Extension: {len(feat)} to {length}')
#         plt.xlabel('Index')
#         plt.ylabel('Value')
#         plt.legend()
#         plt.grid(True)
#
#         if not os.path.exists('processing_features'):
#             os.makedirs('processing_features')
#
#         plt.savefig(f'processing_features/feature_extension_{len(feat)}to{length}.png')
#         plt.close()
#
#         print(f"Visualization saved as 'processing_features/feature_extension_{len(feat)}to{length}.png'")
#
#     return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def modelsize(model, input, type_size=4):
    # check GPU utilisation
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def save_best_record(test_info, file_path, metrics):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(metrics + ": " +str(test_info[metrics][-1]))
    fo.close()


def vid_name_to_path(vid_name, mode):  # TODO: change absolute paths! (only used by visual codes)
    root_dir = '/home/acsguser/Codes/SwinBERT/datasets/Crime/data/'
    types = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery",
             "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    for t in types:
        if vid_name.startswith(t):
            path = root_dir + t + '/' + vid_name
            return path
    if vid_name.startswith('Normal'):
        if mode == 'train':
            path = root_dir + 'Training_Normal_Videos_Anomaly/' + vid_name
        else:
            path = root_dir + 'Testing_Normal_Videos_Anomaly/' + vid_name
        return path
    raise Exception("Unknown video type!!!")


def seed_everything(seed=4869):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_rgb_list_file(ds, is_test):
    if "ucf" in ds:
        ds_name = "Crime"
        if is_test:
            rgb_list_file = 'list/ucf-i3d-test.list'
        else:
            rgb_list_file = 'list/ucf-i3d.list'
    elif "shanghai" in ds:
        ds_name = "Shanghai"
        if is_test:
            rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
        else:
            rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
    elif "violence" in ds:
        ds_name = "Violence"
        if is_test:
            rgb_list_file = 'list/violence-i3d-test.list'
        else:
            rgb_list_file = 'list/violence-i3d.list'
    elif "ped2" in ds:
        ds_name = "UCSDped2"
        if is_test:
            rgb_list_file = 'list/ped2-i3d-test.list'
        else:
            rgb_list_file = 'list/ped2-i3d.list'
    elif "TE2" in ds:
        ds_name = "TE2"
        if is_test:
            rgb_list_file = 'list/te2-i3d-test.list'
        else:
            rgb_list_file = 'list/te2-i3d.list'

    elif "ave" in ds:
        ds_name = "ave"
        if is_test:
            rgb_list_file = 'list/ave-i3d-test.list'
        else:
            rgb_list_file = 'list/ave-i3d.list'

    elif "street" in ds:
        ds_name = "street"
        if is_test:
            rgb_list_file = 'list/street-i3d-test.list'
        else:
            rgb_list_file = 'list/street-i3d.list'

    elif "combine" in ds:
        ds_name = "combine"
        if is_test:
            rgb_list_file = 'list/combined-ped2-sh2-test.list'
        else:
            rgb_list_file = 'list/combined-ped2-sh2.list'

    else:
        raise ValueError("dataset should be either ucf, shanghai, or violence")
    return ds_name, rgb_list_file

def get_rgb_list_file(ds, is_test):
    if "ucf" in ds:
        ds_name = "Crime"
        if is_test:
            rgb_list_file = 'list/ucf-i3d-test.list'
        else:
            rgb_list_file = 'list/ucf-i3d.list'
    elif "shanghai" in ds:
        ds_name = "Shanghai"
        if is_test:
            rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
        else:
            rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
    elif "violence" in ds:
        ds_name = "Violence"
        if is_test:
            rgb_list_file = 'list/violence-i3d-test.list'
        else:
            rgb_list_file = 'list/violence-i3d.list'
    elif "ped2" in ds:
        ds_name = "UCSDped2"
        if is_test:
            rgb_list_file = 'list/ped2-i3d-test.list'
        else:
            rgb_list_file = 'list/ped2-i3d.list'
    elif "TE2" in ds:
        ds_name = "TE2"
        if is_test:
            rgb_list_file = 'list/te2-i3d-test.list'
        else:
            rgb_list_file = 'list/te2-i3d.list'

    elif "ave" in ds:
        ds_name = "ave"
        if is_test:
            rgb_list_file = 'list/ave-i3d-test.list'
        else:
            rgb_list_file = 'list/ave-i3d.list'

    elif "street" in ds:
        ds_name = "street"
        if is_test:
            rgb_list_file = 'list/street-i3d-test.list'
        else:
            rgb_list_file = 'list/street-i3d.list'

    elif "combine" in ds:
        ds_name = "combine"
        if is_test:
            rgb_list_file = 'list/combined-ped2-sh2-test.list'
        else:
            rgb_list_file = 'list/combined-ped2-sh2.list'

    else:
        raise ValueError("dataset should be either ucf, shanghai, or violence")
    return ds_name, rgb_list_file

def get_gt(ds, gt_file):
    if gt_file is not None:
        gt = np.load(gt_file)
    else:
        if 'shanghai' in ds:
            gt = np.load('list/compressed_gt-sh2.npy')
        elif 'ucf' in ds:
            gt = np.load('list/compressed_gt-ucf.npy')
        elif 'violence' in ds:
            gt = np.load('list/compressed_gt-violence.npy')
        elif 'ped2' in ds:
            gt = np.load('list/gt-ped2.npy')
        elif 'TE2' in ds:
            gt = np.load('list/gt-te2.npy')
        elif 'ave' in ds:
            gt = np.load('list/gt-ave.npy')
        elif 'street' in ds:
            gt = np.load('list/gt-street.npy')
        elif 'combine' in ds:
            gt = np.load('list/gt-combine.npy')
        else:
            raise Exception("Dataset undefined!!!")
    return gt




#-----------------------------------------------------------------------------------------------------------------------------------------#

import torch

def squeeze_gt(gt):
    lst = []
    for index in range(0, len(gt), 16):
        lst.append(gt[index])
    return lst

# def repeat_padding(feature):
#     # repeat the sequence of features from the start for multiple times to reach 32 segments
#     batch, num_seg, feature_dim = feature.shape
#     num_repeats = 32 // num_seg
#     remainder = 32 % num_seg
#     padded_feature = feature.repeat(1, num_repeats, 1)
#     if remainder > 0:
#         padded_feature = torch.cat([padded_feature, feature[:, :remainder]], dim=1)
#     return padded_feature
#
# def shift_back_padding(feature):
#     # shift back the number of missing segments to fill out the number of 32 segments
#     #   (e.g. if there is 51 frames, we should need extra 13 frames to pad, so we make it 64 by, first leave 0-31 as it is,
#     #    then for the rest 19 segments, we use segment 18-31 as paddings and add that to the front of the left 19 frames to make it 32)
#     #    but there remains one problem is that how do we know which parts are paddings, when we need to compare the origianl number of segments
#     #    to compare to the ground truth and leave out the paddings?
#     batch, num_seg, feature_dim = feature.shape
#     padding_size = 32 - num_seg
#     padded_feature = torch.cat([feature[:, -padding_size:], feature], dim=1)
#     return padded_feature
#
# def test_padding(feature):
#     batch, seg, _ = feature.shape
#     if seg < 32:
#         feature = repeat_padding(feature)
#     if seg > 32 & seg % 32 != 0:
#         feature = shift_back_padding(feature)
#     else:
#         feature = feature
#     return feature

def repeat_padding(feature, mask):
    batch, num_seg, feature_dim = feature.shape
    num_repeats = 32 // num_seg
    padded_feature = feature.repeat(1, num_repeats + 1, 1)
    padded_mask = mask.repeat(1, num_repeats + 1)

    padded_feature = padded_feature[:, :32, :]
    padded_mask = padded_mask[:, :32]

    padded_mask[:, num_seg:] = True

    return padded_feature, padded_mask

def shift_back_padding(feature, mask):
    batch, num_seg, feature_dim = feature.shape
    padding_size = 32 - (num_seg % 32)
    num_windows = num_seg // 32 + 1
    total_size = num_windows * 32

    padded_feature = torch.zeros(batch, total_size, feature_dim, device=feature.device)
    padded_mask = torch.zeros(batch, total_size, dtype=torch.bool, device=mask.device)

    padded_feature[:, :num_seg] = feature
    padded_mask[:, :num_seg] = mask

    start_index = (num_windows - 1) * 32

    end_index_for_padding = num_seg - padding_size

    repeated_segments = feature[:, end_index_for_padding:num_seg]
    padded_feature[:, start_index:start_index + padding_size] = repeated_segments

    padded_mask[:, start_index:start_index + padding_size] = True

    return padded_feature, padded_mask

def test_padding(feature):
    print("feature to be padded: ", feature.shape)
    batch, seg, _ = feature.shape
    mask = torch.zeros(batch, seg, dtype=torch.bool)
    if seg < 32:
        feature, mask = repeat_padding(feature, mask)
    elif seg > 32 and seg % 32 != 0:
        print("shift back padding is used")
        feature, mask = shift_back_padding(feature, mask)
    return feature, mask



def compress_sequence(visual_features, text_features, labels, target_length=32):
    batch_size, num_crops, seq_length, visual_feature_dim = visual_features.shape
    _, _, _, text_feature_dim = text_features.shape
    unique_labels, inverse_indices = torch.unique(torch.tensor(labels), return_inverse=True)
    num_unique_labels = len(unique_labels)

    compressed_visual_features = torch.zeros(batch_size, num_crops, target_length, visual_feature_dim,
                                             device=visual_features.device)
    compressed_text_features = torch.zeros(batch_size, num_crops, target_length, text_feature_dim,
                                           device=text_features.device)
    compressed_labels = torch.zeros(target_length, dtype=torch.long, device=visual_features.device)

    for i in range(num_unique_labels):
        label_indices = (inverse_indices == i).nonzero().squeeze(-1)
        if label_indices.numel() > 0:
            label_indices = label_indices.to(
                visual_features.device)  # Move label_indices to the same device as visual_features
            label_visual_features = visual_features[:, :, label_indices].reshape(-1, label_indices.numel(),
                                                                                 visual_feature_dim)
            label_text_features = text_features[:, :, label_indices].reshape(-1, label_indices.numel(),
                                                                             text_feature_dim)
            compressed_visual_features[:, :, i] = label_visual_features.mean(dim=1)
            compressed_text_features[:, :, i] = label_text_features.mean(dim=1)
            compressed_labels[i] = unique_labels[i]

    return compressed_visual_features, compressed_text_features, compressed_labels


def get_cos_matrix(feature):

    n_feature = F.normalize(feature, p=2, dim=1)
    # print("input_tensor: ", input_tensor)
    # nancheck_input = torch.isnan(input_tensor)
    # print("nancheck for input cosine", torch.all(nancheck_input == False))

    # Proceed with the cosine similarity calculation
    # dot_product = torch.mm(input_tensor, input_tensor.t())
    # print("dot_product: ", dot_product)
    # col_norms = input_tensor.norm(dim=1)
    # col_norms = F.normalize(input_tensor, p=2, dim=1)
    # print("row_norms: ", row_norms.shape)
    # row_norms = row_norms.unsqueeze(1)
    # print("row_norms: ", row_norms.shape)
    adjacency_matrix = torch.mm(n_feature, n_feature.t())
    # print("Adjacency matrix: ", adjacency_matrix)
    adjacency_matrix = adjacency_matrix.fill_diagonal_(1)
    adjacency_matrix[adjacency_matrix < 0.6] = 0
    # print("adjacency_matrix: ", adjacency_matrix.shape)

    # adjacency_matrix_normalised = F.normalize(adjacency_matrix, p=2, dim=1)

    return adjacency_matrix

# def get_cos_matrix(input_tensor):
#     epsilon = 1e-9  # Small positive value to avoid division by zero
#
#     # Check for zero vectors
#     is_zero_vector = torch.all(input_tensor == 0, dim=1)
#
#     # Replace zero vectors with a small positive value
#     input_tensor[is_zero_vector] = epsilon
#
#     normalized_data = F.normalize(input_tensor, p=2, dim=1)
#
#     # Set the rows and columns corresponding to zero vectors to zero
#     normalized_data[is_zero_vector] = 0
#
#     adjacency_matrix = torch.mm(normalized_data, normalized_data.t())
#     adjacency_matrix[adjacency_matrix < 0.8] = epsilon
#
#     # Set the diagonal elements corresponding to zero vectors to 1
#     adjacency_matrix[is_zero_vector, :] = 0
#     adjacency_matrix[:, is_zero_vector] = 0
#     torch.diagonal(adjacency_matrix, dim1=0, dim2=1)[is_zero_vector] = 1
#
#     return adjacency_matrix


def get_deg_matrix(adj_matrix):
    degree_sequence = torch.count_nonzero(adj_matrix, dim=1)
    degree_matrix = torch.diag(degree_sequence)
    # nancheck_deg = torch.isnan(degree_matrix)
    # print("nancheck for input cosine", torch.all(nancheck_deg == False))
    return degree_matrix


def get_normalised_adj_matrix(adj_matrix):
    """
    Calculate the normalized adjacency matrix.

    """

    adj_matrix = adj_matrix.to(torch.float64)

    # print("Original adj matrix", adj_matrix)

    I = torch.eye(32, device='cuda')

    A = adj_matrix + I

    # print("identity matrix", I)
    #
    # print("new adj matrix", A)

    # nancheck_adj = torch.isnan(A)
    # print("normalise adj input check", torch.all(nancheck_adj == False))


    deg_matrix = get_deg_matrix(A)

    sqrt_deg_matrix_inv = deg_matrix.clone()
    sqrt_deg_matrix_inv = sqrt_deg_matrix_inv.float()

    sqrt_deg_matrix_inv = sqrt_deg_matrix_inv.to(torch.float64)

    # print("before", sqrt_deg_matrix_inv)
    # Calculate the power of -0.5 only for the diagonal elements
    torch.diagonal(sqrt_deg_matrix_inv, dim1=0, dim2=1).pow_(-0.5)

    # print("after", sqrt_deg_matrix_inv)


    # print("deg matrix", deg_matrix)


    sqrt_deg_matrix_inv[sqrt_deg_matrix_inv == float('inf')] = 0
    sqrt_deg_matrix_inv[sqrt_deg_matrix_inv == float('nan')] = 0

    # nancheck_deg_inv = torch.isnan(sqrt_deg_matrix_inv)
    # print("nancheck for deg inv", torch.all(nancheck_deg_inv == False))

    # print("inverse deg matrix", sqrt_deg_matrix_inv)

    S = torch.mm(torch.mm(sqrt_deg_matrix_inv, A), sqrt_deg_matrix_inv)
    # print("S" ,S)

    # S_normalised = F.normalize(S,p=2,dim=1)

    return S


def get_updated_feature(S, features,original_features, layer=10, alpha=0.1):
    """
    Feature update step using the similarity matrix and existing features.

    """
    S = S.to(torch.float64)
    features = features.to(torch.float64)

    features = F.normalize(features, p=2, dim=1)

    # print("feature input for update", features)
    #
    # nancheck_original = torch.isnan(original_features)
    # print("nancheck for original", torch.all(nancheck_original == False))
    #
    # nancheck = torch.isnan(features)
    # print("nancheck for updated input feature", torch.all(nancheck == False))

    # power_feature = torch.pow(features, layer)
    # nancheck_power = torch.isnan(power_feature)
    # print("nancheck power", torch.all(nancheck_power == False))
    #
    # print("power feature", power_feature)

    # new_features = (1 - alpha) * torch.mm(S, power_feature) + alpha * original_features
    new_features = torch.mm(S, features)

    # nancheck_updated = torch.isnan(new_features)
    # print("nancheck updated generated", torch.all(nancheck_updated == False))

    # print("new_features", new_features)

    # print(new_features)

    new_features_normalised = F.normalize(new_features, p=2, dim=1)

    return new_features_normalised

def get_uppertriangular_matrices(matrices):
    # (batch_size, 32, 32)
    # print("upper check", matrices.shape)
    row = matrices.shape[1]
    col = matrices.shape[2]
    indices = torch.triu_indices(32, 32, offset=0)
    upper_triangular_matrices = matrices[:, indices[0], indices[1]]
    return upper_triangular_matrices





