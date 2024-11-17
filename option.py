import argparse

parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
# parser.add_argument('--rgb-list', default='list/shanghai-i3d-train-10crop.list', help='list of rgb features ')
parser.add_argument('--test-rgb-list', default=None, help='list of test rgb features ')
parser.add_argument('--gt', default=None, help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--model-name', default='rtfm', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--dataset', default='ped2', help='dataset to train on (shanghai, ucf, ped2, violence, TE2)')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')


parser.add_argument('--ablation', default='w_var', help='ablation note')
parser.add_argument('--go', default='matrix', help='graph operation type')
parser.add_argument('--epoch', type=int, default=50, help='graph operation type')
parser.add_argument('--wandb', type=bool, default=False, help='graph operation type')
parser.add_argument('--lr', type=float, default=0.001, help='learning rates for steps(list form)')
parser.add_argument('--wd', type=float, default=1e-3, help='learning rates for steps(list form)')
parser.add_argument('--m', type=int, default=4, help='graph operator dimension')
parser.add_argument('--n', type=int, default=4, help='graph operator dimension')
parser.add_argument('--modal', default='all', help='modality')









parser.add_argument('--seed', type=int, default=4869, help='random seed (default: 4869)')
parser.add_argument('--max-epoch', type=int, default=1000, help='maximum iteration to train (default: 1000)')
parser.add_argument('--feature-group', default='both', choices=['both', 'vis', 'text'], help='feature groups used for the model')
parser.add_argument('--fusion', type=str, default='concat', help='how to fuse vis and text features')
parser.add_argument('--normal_weight', type=float, default=1, help='weight for normal loss weights')
parser.add_argument('--abnormal_weight', type=float, default=1, help='weight for abnormal loss weights')
parser.add_argument('--aggregate_text', action='store_true', default=False, help='whether to aggregate text features')
parser.add_argument('--extra_loss', action='store_true', default=False, help='whether to use extra loss')
parser.add_argument('--save_test_results', action='store_true', default=False, help='whether to save test results')
parser.add_argument('--alpha', type=float, default=1.0, help='weight for RTFM loss')


parser.add_argument('--emb_folder', type=str, default='sent_emb_n', help='folder for text embeddings, used to differenciate different swinbert pretrained models')
parser.add_argument('--c3d_folder', type=str, default='c3d', help='folder for text embeddings, used to differenciate different swinbert pretrained models')
parser.add_argument('--swin_folder', type=str, default='swin', help='folder for text embeddings, used to differenciate different swinbert pretrained models')
parser.add_argument('--pos_folder', type=str, default='pos', help='folder for text embeddings, used to differenciate different swinbert pretrained models')




parser.add_argument('--emb_dim', type=int, default=768, help='dimension of text embeddings')