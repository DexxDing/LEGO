
from utils import *
from dataset import Dataset
from utils import seed_everything
import option
from config import Config
from torch.utils.data import  DataLoader
import torch
import graphfusion
from test_10crop import test


if __name__ == '__main__':
    torch.manual_seed(4869)
    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=False,
                               generator=torch.Generator(device='cuda'))
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=False,
                               generator=torch.Generator(device='cuda'))

    test_loader = DataLoader(Dataset(args, test_mode=True), batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=False)

    viz_name = 'BNGF'
    viz = Visualizer(env=viz_name, use_incoming_socket=False)


    model = graphfusion.GraphClassifiction(batch_size=32, m=5, n=5)

    best_model_path = "./ckpt/model_epoch_75_0.0841.pth"

    model_state_dict = torch.load(best_model_path)

    model.load_state_dict(model_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epoch = 75

    test_auc, test_ap = test(test_loader, model, args, device, epoch)

    print(f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")

