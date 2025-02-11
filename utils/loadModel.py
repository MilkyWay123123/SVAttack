import os.path
import sys

sys.path.append("..")

import torch
import torch.nn as nn
from models.stgcn.st_gcn import STGCN_Model
from models.agcn.agcn import Model as AGCN_Model

def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')


def loadSTGCN():
    load_path = './ModelWeights/STGCN/ntu60/ntu-xsub.pt'
    print(f'load STGCN:{load_path}')
    graph_args = {
        "layout": "ntu-rgb+d",
        "strategy": "spatial"
    }
    stgcn = STGCN_Model(3, 60, graph_args, True)
    stgcn.eval()
    pretrained_weights = torch.load(load_path)
    stgcn.load_state_dict(pretrained_weights)
    stgcn.cuda()

    return stgcn


def loadAGCN():
    load_path = '/23085412008/模型权重/AGCN/ntu60/ntu_cs_agcn_joint-49-31500.pt'
    print(f'load agcn:{load_path}')
    agcn = AGCN_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    agcn.eval()
    pretrained_weights = torch.load(load_path)
    agcn.load_state_dict(pretrained_weights)
    agcn.cuda()

    return agcn

def getModel(AttackedModel):
    if AttackedModel == 'stgcn':
        model = loadSTGCN()
    elif AttackedModel == 'agcn':
        model = loadAGCN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total number of model parameters：{total_params:.2f} M")
    return model
