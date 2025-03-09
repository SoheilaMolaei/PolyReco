import math
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
import dgl
from rdkit import Chem
from sklearn.impute import SimpleImputer
from helpers import (
    get_hard_soft_value,
    feature_size,
    graph_adjacency
)

def DataExtN():
    df = pd.read_excel('database.xlsx', 'BCPs')
    dfH = pd.read_excel('database.xlsx', 'Homopolymers')
    print('- Total BCP data:', df.shape[0])
    print('- Total Homopolymer data:', dfH.shape[0])
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    XE = [df['σbreak (MPa)']]
    # XE1 = [df['εbreak (%)']]
    # XE = [df['εbreak (%)']]
    XE1 = [df['εbreak (%)']]

    smiles = df['Big_Smile'].tolist()
    counts = df['Block DP']
    features, features1, features2 = [], [], []
    G = nx.DiGraph()
    
    # Extract features for each SMILES entry
    for i in range(len(smiles)):
        features.append([XE[0][i]])
        features1.append([XE[0][i]])
        features2.append([XE1[0][i]])
    features = imputer.fit_transform(features)
    
    Count = 0
    for i in range(len(smiles)):
        Sp = smiles[i][1:-1].split('}{')
        Cn = str(counts[i]).split(':')
        if len(Cn[0].split('/')) == 1:
            Cn[0] = str(2 * float(Cn[0]))
        else:
            AC = Cn[0].split('/')
            Cn[0] = str(2 * float(AC[0])) + '/' + str(2 * float(AC[1]))
        
        # Expecting two segments
        Sp = [Sp[0], Sp[1]]
        Spp = []
        for l in range(2):
            if len(Sp[l].split(',')) == 1:
                Spp.append(Sp[l])
                
        data = [Chem.MolFromSmiles(line) for line in Spp if Chem.MolFromSmiles(line) is not None]
        if not data:
            continue  # Skip if no valid molecule is generated
        atom_labels = sorted(set([atom.GetAtomicNum() for mol in data for atom in mol.GetAtoms()] + [0]))
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType() for mol in data for bond in mol.GetBonds())))
        bond_encoder_m = {l: ii for ii, l in enumerate(bond_labels)}
    
        Feat = []
        for idx, mol in enumerate(data):
            combined_feature = torch.cat([feature_size(mol, atom_labels),
                                          graph_adjacency(mol, bond_encoder=bond_encoder_m)], dim=1)
            DP_values = Cn[idx].split('/')
            dp_weight = torch.sqrt(torch.tensor(float(DP_values[0]), dtype=torch.float))
            weighted_features = (combined_feature + 1) * dp_weight
            Feat.append(weighted_features.float())
    
        Feat = torch.stack(Feat)
        Feat = F.pad(Feat, (0, 65 - Feat.size(-1)), "constant", 0)
    
        # Process sub-features if any segment has a comma
        if len(Sp[0].split(',')) > 1 or len(Sp[1].split(',')) > 1:
            if len(Sp[0].split(',')) > 1:
                ss = Sp[0].split(',')
                jj = Cn[0].split('/')
            if len(Sp[1].split(',')) > 1:
                ss = Sp[1].split(',')
                jj = Cn[1].split('/')
            data = [Chem.MolFromSmiles(ss[0])]
            atom_labels = sorted(set([atom.GetAtomicNum() for mol in data for atom in mol.GetAtoms()] + [0]))
            bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType() for mol in data for bond in mol.GetBonds())))
            bond_encoder_m = {l: ii for ii, l in enumerate(bond_labels)}
            Feat1 = []
            for idx, mol in enumerate(data):
                combined_feature = torch.cat([feature_size(mol, atom_labels),
                                              graph_adjacency(mol, bond_encoder=bond_encoder_m)], dim=1)
                dp_weight = torch.sqrt(torch.tensor(float(jj[0]), dtype=torch.float))
                weighted_features = (combined_feature + 1) * dp_weight
                Feat1.append(weighted_features.float())
            Feat1 = torch.stack(Feat1)
            Feat1 = F.pad(Feat1, (0, 65 - Feat1.size(-1)), "constant", 0)
    
            data = [Chem.MolFromSmiles(ss[1])]
            atom_labels = sorted(set([atom.GetAtomicNum() for mol in data for atom in mol.GetAtoms()] + [0]))
            bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType() for mol in data for bond in mol.GetBonds())))
            bond_encoder_m = {l: ii for ii, l in enumerate(bond_labels)}
            Feat2 = []
            for idx, mol in enumerate(data):
                combined_feature = torch.cat([feature_size(mol, atom_labels),
                                              graph_adjacency(mol, bond_encoder=bond_encoder_m)], dim=1)
                dp_weight = torch.sqrt(torch.tensor(float(jj[1]), dtype=torch.float))
                weighted_features = (combined_feature + 1) * dp_weight
                Feat2.append(weighted_features.float())
            Feat2 = torch.stack(Feat2)
            Feat2 = F.pad(Feat2, (0, 65 - Feat2.size(-1)), "constant", 0)
            Feat3 = Feat1 + Feat2
            Feat = torch.cat((Feat, Feat3), dim=0)
    
        # Create padded features for the node (flatten and pad to fixed length 4000)
        padded_features = []
        for feat in Feat:
            feat_flat = feat.reshape(-1)
            pad_len = 4000 - feat_flat.shape[0]
            feat_flat = torch.cat([feat_flat, torch.zeros(pad_len)])
            padded_features.append(feat_flat)
        padded_features = torch.stack(padded_features)
        # --- FIX: Ensure we have at least two feature vectors ---
        if padded_features.shape[0] < 2:
            padded_features = torch.cat([padded_features, padded_features], dim=0)
    
        Feat_T = torch.tensor(features[i])
        Feat_TE = torch.tensor(features1[i])
        Feat_TE1 = torch.tensor(features2[i])
    
        value0 = get_hard_soft_value(Sp[0], dfH)
        value1 = get_hard_soft_value(Sp[1], dfH)
    
        node_name0 = Sp[0] + ';' + Cn[0]
        node_name1 = Sp[1] + ';' + (Cn[1] if len(Cn) > 1 else Cn[0])
    
        if value0 is not None and value1 is None:
            if not G.has_node(node_name0):
                Count += 1
                G.add_node(node_name0, Feature=padded_features[0],
                           weight=Feat_T, bipartite=value0,
                           weightBreak=Feat_TE1, name=Count)
            if not G.has_node(node_name1):
                Count += 1
                G.add_node(node_name1, Feature=padded_features[1],
                           weight=Feat_T, bipartite=1 - value0,
                           weightBreak=Feat_TE1, name=Count)
        elif value1 is not None and value0 is None:
            if not G.has_node(node_name0):
                Count += 1
                G.add_node(node_name0, Feature=padded_features[0],
                           weight=Feat_T, bipartite=1 - value1,
                           weightBreak=Feat_TE1, name=Count)
            if not G.has_node(node_name1):
                Count += 1
                G.add_node(node_name1, Feature=padded_features[1],
                           weight=Feat_T, bipartite=value1,
                           weightBreak=Feat_TE1, name=Count)
        elif value0 is not None and value1 is not None:
            if not G.has_node(node_name0):
                Count += 1
                G.add_node(node_name0, Feature=padded_features[0],
                           weight=Feat_T, bipartite=value0,
                           weightBreak=Feat_TE1, name=Count)
            if not G.has_node(node_name1):
                Count += 1
                G.add_node(node_name1, Feature=padded_features[1],
                           weight=Feat_T, bipartite=value1,
                           weightBreak=Feat_TE1, name=Count)
        else:
            if not G.has_node(node_name0):
                Count += 1
                G.add_node(node_name0, Feature=padded_features[0],
                           weight=Feat_T, bipartite=0,
                           weightBreak=Feat_TE1, name=Count)
            if not G.has_node(node_name1):
                Count += 1
                G.add_node(node_name1, Feature=padded_features[1],
                           weight=Feat_T, bipartite=1,
                           weightBreak=Feat_TE1, name=Count)
    
        G.add_edge(node_name0, node_name1, Feature=padded_features[0],
                   weight=Feat_TE, weightBreak=Feat_TE1)
    
    mapping = {node: i for i, node in enumerate(G.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    dgl_g = dgl.from_networkx(G, node_attrs=['Feature','weight','weightBreak','bipartite','name'],
                              edge_attrs=['Feature','weight','weightBreak'])
    return dgl_g, G, reverse_mapping
