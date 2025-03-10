# utils.py
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem

def get_hard_soft_value(smile: str, dataframe: pd.DataFrame):
    """
    Returns the 'Hard/Soft' value for a given SMILES string.
    """
    filtered_df = dataframe[dataframe['Big_Smile'] == smile]
    return filtered_df.iloc[0]['Hard/Soft'] if not filtered_df.empty else None

def feature_size(mol, atom_labels, max_atoms=60):
    """
    Creates a one-hot feature matrix for a molecule.
    """
    features = torch.zeros((max_atoms, len(atom_labels)))
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_atoms:
            break
        features[i, atom_labels.index(atom.GetAtomicNum())] = 1
    return features

def graph_adjacency(mol, max_atoms=60, bond_encoder=None):
    """
    Creates an adjacency matrix for a molecule using a bond encoder.
    """
    bond_encoder = bond_encoder or {}
    adjacency = torch.zeros((max_atoms, max_atoms))
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start < max_atoms and end < max_atoms:
            adjacency[start, end] = adjacency[end, start] = bond_encoder.get(bond.GetBondType(), 0)
    return adjacency

def remove_unique_classes(class_matrix, feature_matrix):
    """
    Remove entries corresponding to unique classes.
    """
    unique_classes, class_counts = np.unique(class_matrix, axis=0, return_counts=True)
    unique_indices = np.where(class_counts == 1)[0]
    return np.delete(class_matrix, unique_indices, axis=0), np.delete(feature_matrix, unique_indices, axis=0)

def map_all_edges_g_to_G(g, G):
    """
    Maps edges in the DGL graph 'g' back to the original NetworkX graph 'G' 
    based on the 'name' attribute.
    """
    edge_to_str_name_mapping = {}
    for edge_id in range(g.number_of_edges()):
        src_id, dst_id = g.find_edges(edge_id)
        src_name_in_g = g.ndata['name'][src_id].item()
        dst_name_in_g = g.ndata['name'][dst_id].item()
        src_name_in_G = dst_name_in_G = None
        for node, data in G.nodes(data=True):
            if data.get('name') == src_name_in_g:
                src_name_in_G = node
            if data.get('name') == dst_name_in_g:
                dst_name_in_G = node
            if src_name_in_G is not None and dst_name_in_G is not None:
                break
        if src_name_in_G is not None and dst_name_in_G is not None:
            edge_to_str_name_mapping[edge_id] = (src_name_in_G, dst_name_in_G)
    return edge_to_str_name_mapping

