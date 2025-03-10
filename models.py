# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = dgl.nn.SAGEConv(in_feats, h_feats, 'mean')
        self.dropout = nn.Dropout(0.5)
        self.conv2 = dgl.nn.SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = dgl.nn.SAGEConv(h_feats, h_feats, 'mean')
        self.conv4 = dgl.nn.SAGEConv(h_feats, h_feats, 'mean')
        self.conv5 = dgl.nn.SAGEConv(h_feats, h_feats, 'mean')

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
        h = self.conv5(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv4(g, h)
        return h

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super(MLPPredictor, self).__init__()
        self.linear1 = nn.Linear(h_feats * 2, h_feats)
        self.linear2 = nn.Linear(h_feats, 1)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self._apply_edges)
            return g.edata['score'], g

    def _apply_edges(self, edges):
        h_cat = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        score = F.relu(self.linear1(h_cat))
        score = self.linear2(score)
        return {'score': score.squeeze(1)}

def init_weights(m):
    """
    Initialize weights for Linear and SAGEConv layers.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif hasattr(m, 'fc_self'):
        torch.nn.init.xavier_uniform_(m.fc_self.weight)
        if m.fc_self.bias is not None:
            m.fc_self.bias.data.fill_(0.01)
    elif hasattr(m, 'fc_neigh'):
        torch.nn.init.xavier_uniform_(m.fc_neigh.weight)
        if m.fc_neigh.bias is not None:
            m.fc_neigh.bias.data.fill_(0.01)

class GraphDecoder(nn.Module):
    def __init__(self, h_feats, out_feats):
        super(GraphDecoder, self).__init__()
        self.linear1 = nn.Linear(h_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, out_feats)

    def forward(self, h):
        x = F.relu(self.linear1(h))
        return self.linear2(x)

