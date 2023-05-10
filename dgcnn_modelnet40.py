import torch.nn as nn
import torch.nn.functional as F
from dgcnn.pytorch.model import DGCNN as DGCNN_original
#from all_utils import DATASET_NUM_CLASS


def min_max_normalize(pc):
    # pc : B x N (=1024) x C (=3)
    assert pc.shape[2] == 3
    x = pc[:, :, 0]
    y = pc[:, :, 1]
    z = pc[:, :, 2]
    new_pc = pc.detach().clone()
    new_pc -= (pc.min(dim=1, keepdim=True).values + pc.max(dim=1, keepdim=True).values) / 2
    length = (pc.max(dim=1, keepdim=True).values -
            pc.min(dim=1,keepdim=True).values).max(dim=2, keepdim=True).values
    new_pc *= 2.0 / length

    return new_pc

class DGCNN(nn.Module):

    def __init__(self, task='cls', dataset='modelnet40'):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = 40 #DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.k = 20
                    self.emb_dims = 1024
                    self.dropout = 0.5
                    self.leaky_relu = 1
            args = Args()
            self.model = DGCNN_original(args, output_channels=num_classes)
            #self.load_state_dict(torch.load('outputs/dgcnn_modelnet40_best_test.pth')['model_state'])

        else:
            assert False

    def forward(self, pc): #, cls=None):
        # normalize pc here!
        #pc = min_max_normalize(pc)
        #pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        logit = self.model(pc)
        out = {'cls': logit}
        return out
