import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = dgl.nn.SAGEConv(in_feats, h_feats, 'mean')
        self.dropout = nn.Dropout(0.5)  # Dropout rate 0.5
        self.conv2 = dgl.nn.SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = dgl.nn.SAGEConv(h_feats, h_feats, 'mean')
        self.conv4 = dgl.nn.SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv4(g, h)
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        g.ndata['h'] = h
        g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        # Return scores as a 1D tensor along with the updated graph
        return g.edata['score'][:, 0], g
