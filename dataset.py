import pdb

import torch.utils.data as data
import numpy as np
from utils import process_feat, get_rgb_list_file
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.emb_folder = args.emb_folder
        self.c3d_folder = args.c3d_folder
        self.swin_folder = args.swin_folder
        self.pose_folder = args.pos_folder
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.feature_size = args.feature_size
        if args.test_rgb_list is None:
            _, self.rgb_list_file = get_rgb_list_file(args.dataset, test_mode)
        else:
            self.rgb_list_file = args.test_rgb_list

        # deal with different I3D feature version
        if 'v2' in self.dataset:
            self.feat_ver = 'v2'
        elif 'v3' in self.dataset:
            self.feat_ver = 'v3'
        else:
            self.feat_ver = 'v1'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:  # list for training would need to be ordered from normal to abnormal
            if 'shanghai' in self.dataset:
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
            elif 'ucf' in self.dataset:
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
            elif 'violence' in self.dataset:
                if self.is_normal:
                    self.list = self.list[1904:]
                    print('normal list for violence')
                else:
                    self.list = self.list[:1904]
                    print('abnormal list for violence')
            elif 'ped2' in self.dataset:
                if self.is_normal:
                    self.list = self.list[7:]
                    print('normal list for ped2', len(self.list))
                    print('normal list', self.list)
                else:
                    self.list = self.list[:7]
                    print('abnormal list for ped2', len(self.list))
                    print('abnormal list', self.list)
            elif 'TE2' in self.dataset:  # 注意index从0开始，而pycharm行号从1开始
                if self.is_normal:
                    self.list = self.list[23:]
                    print('normal list for TE2', len(self.list))
                else:
                    self.list = self.list[:23]
                    print('abnormal list for TE2', len(self.list))
            elif 'ave' in self.dataset:
                if self.is_normal:
                    self.list = self.list[15:]
                    print('normal list for ave', len(self.list))
                    print('normal list', self.list)
                else:
                    self.list = self.list[:15]
                    print('abnormal list for ave', len(self.list))
                    print('abnormal list', self.list)
            elif 'street' in self.dataset:
                if self.is_normal:
                    self.list = self.list[24:]
                    print('normal list for street', len(self.list))
                    print('normal list', self.list)
                else:
                    self.list = self.list[:24]
                    print('abnormal list for street', len(self.list))
                    print('abnormal list', self.list)
            elif 'combine' in self.dataset:
                if self.is_normal:
                    self.list = self.list[70:]
                    print('normal list for combine', len(self.list))
                    print('normal list', self.list)
                else:
                    self.list = self.list[:70]
                    print('abnormal list for combine', len(self.list))
                    print('abnormal list', self.list)
            else:
                raise Exception("Dataset undefined!!!")

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        i3d_path = self.list[index].strip('\n')
        c3d_path = self.list[index].strip('\n')
        # print(i3d_path)

        if self.feat_ver == 'v2':
            i3d_path = i3d_path.replace('i3d_v1', 'i3d_v2')
        elif self.feat_ver == 'v3':
            i3d_path = i3d_path.replace('i3d_v1', 'i3d_v3')

        # print(i3d_path)

        features = np.load(i3d_path, allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if 'ucf' in self.dataset:
            text_path = "save/Crime/" + self.emb_folder + "/" + i3d_path.split("/")[-1][:-7]+"emb.npy"
        elif 'shanghai' in self.dataset:
            text_path = "save/Shanghai/" + self.emb_folder + "/" + i3d_path.split("/")[-1][:-7]+"emb.npy"
            c3d_path = "save/Shanghai/" + self.c3d_folder + "/" + i3d_path.split("/")[-1][:-7]+"c3d.npy"
            swin_path = "save/Shanghai/" + self.swin_folder + "/" + i3d_path.split("/")[-1][:-7]+"swin.npy"
            pose_path = "save/Shanghai/" + self.pose_folder + "/" + i3d_path.split("/")[-1][:-7]+"pose.npy"
        elif 'violence' in self.dataset:
            text_path = "save/Violence/" + self.emb_folder + "/" + i3d_path.split("/")[-1][:-7]+"emb.npy"
        elif 'ped2' in self.dataset:
            text_path = "save/UCSDped2/" + self.emb_folder + "/" + i3d_path.split("/")[-1][:-7]+"emb.npy"
        elif 'TE2' in self.dataset:
            text_path = "save/TE2/" + self.emb_folder + "/" + i3d_path.split("/")[-1][:-7]+"emb.npy"
        elif 'ave' in self.dataset:
            text_path = "save/Avenue/" + self.emb_folder + "/" + i3d_path.split("/")[-1][:-7]+"emb.npy"
        elif 'street' in self.dataset:
            text_path = "save/StreetScene/" + self.emb_folder + "/" + i3d_path.split("/")[-1][:-7]+"emb.npy"
        elif 'combine' in self.dataset:
            text_path = "save/Combine/" + self.emb_folder + "/" + i3d_path.split("/")[-1][:-7]+"emb.npy"

        else:
            raise Exception("Dataset undefined!!!")
        text_features = np.load(text_path, allow_pickle=True)
        text_features = np.array(text_features, dtype=np.float32)  # [snippet no., 768]

        c3d_features = np.load(c3d_path, allow_pickle=True)
        c3d_features = np.array(c3d_features, dtype=np.float32)  # [snippet no., 4096]

        swin_features = np.load(swin_path, allow_pickle=True)
        swin_features = np.array(swin_features, dtype=np.float32)  # [snippet no., 1024]

        pose_features = np.load(pose_path, allow_pickle=True)
        pose_features = np.array(pose_features, dtype=np.float32)  # [snippet no., 2176]

        # assert features.shape[0] == text_features.shape[0]
        if self.feature_size == 1024:
            text_features = np.tile(text_features, (5, 1, 1))  # [10,snippet no.,768]
        elif self.feature_size == 2048:
            text_features = np.tile(text_features, (10, 1, 1))  # [10,snippet no.,768]
            c3d_features = np.tile(c3d_features, (10, 1, 1))
            swin_features = np.tile(swin_features, (10, 1, 1))
            pose_features = np.tile(pose_features, (10, 1, 1))
        else:
            raise Exception("Feature size undefined!!!")






        if self.tranform is not None:
            features = self.tranform(features)

        if self.test_mode:
            text_features = text_features.transpose(1, 0, 2)  # [snippet no.,10,768]
            c3d_features = c3d_features.transpose(1, 0, 2)
            swin_features = swin_features.transpose(1, 0, 2)
            pose_features = pose_features.transpose(1, 0, 2)
            return features, c3d_features, swin_features, pose_features, text_features, 
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [snippet no., 10, 2048] -> [10, snippet no., 2048]
            print('feature dimension', features.shape)
            divided_features = []
            for feature in features:  # loop 10 times
                feature = process_feat(feature, 32)  # divide a video into 32 segments/snippets/clips
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)  # [10,32,2048]

            div_feat_text = []
            for text_feat in text_features:
                text_feat = process_feat(text_feat, 32)  # [32,768]
                div_feat_text.append(text_feat)
            div_feat_text = np.array(div_feat_text, dtype=np.float32)
            assert divided_features.shape[1] == div_feat_text.shape[1], str(self.test_mode) + "\t" + str(divided_features.shape[1]) + "\t" + div_feat_text.shape[1]


            div_c3d_feat = []
            for c3d_feat in c3d_features:
                c3d_feat = process_feat(c3d_feat, 32)  # [32,768]
                div_c3d_feat.append(c3d_feat)
            div_c3d_feat = np.array(div_c3d_feat, dtype=np.float32)
            assert divided_features.shape[1] == div_c3d_feat.shape[1], str(self.test_mode) + "\t" + str(divided_features.shape[1]) + "\t" + div_c3d_feat.shape[1]
        

            div_swin_feat = []
            for swin_feat in swin_features:
                swin_feat = process_feat(swin_feat, 32)  # [32,768]
                div_swin_feat.append(swin_feat)
            div_swin_feat = np.array(div_swin_feat, dtype=np.float32)
            assert divided_features.shape[1] == div_swin_feat.shape[1], str(self.test_mode) + "\t" + str(divided_features.shape[1]) + "\t" + div_swin_feat.shape[1]


            div_pose_feat = []
            for pose_feat in pose_features:
                pose_feat = process_feat(pose_feat, 32)  # [32,768]
                div_pose_feat.append(pose_feat)
            div_pose_feat = np.array(div_pose_feat, dtype=np.float32)
            assert divided_features.shape[1] == div_swin_feat.shape[1], str(self.test_mode) + "\t" + str(divided_features.shape[1]) + "\t" + div_swin_feat.shape[1]


            return divided_features, div_c3d_feat, div_swin_feat, div_pose_feat, div_feat_text, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
