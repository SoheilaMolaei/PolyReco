import numpy as np
import torch
import networkx as nx
import dgl
from rdkit import Chem

def get_hard_soft_value(smile, dataframe):
    """
    Get the 'Hard/Soft' value from the dataframe based on the smile string.
    """
    filtered_df = dataframe[dataframe['Big_Smile'] == smile]
    return filtered_df.iloc[0]['Hard/Soft'] if not filtered_df.empty else None

def graph_features(mol, atom_labels, max_length=None):
    """
    Compute one‐hot features for the atoms in a molecule.
    Pads the feature matrix up to max_length.
    """
    max_length = max_length or mol.GetNumAtoms()
    features = np.array([[a.GetAtomicNum() == i for i in atom_labels] for a in mol.GetAtoms()], dtype=np.int32)
    padding = np.zeros((max_length - features.shape[0], features.shape[1]))
    return np.vstack((features, padding))

def feature_size(mol, atom_labels, max_atoms=60):
    """
    Create a tensor of size (max_atoms x len(atom_labels)) with one-hot encoding for each atom.
    """
    features = torch.zeros((max_atoms, len(atom_labels)))
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_atoms:
            break
        try:
            idx = atom_labels.index(atom.GetAtomicNum())
        except ValueError:
            continue  # In case the atomic number is not in atom_labels.
        features[i, idx] = 1
    return features

def graph_adjacency(mol, max_atoms=60, bond_encoder=None):
    """
    Create an adjacency matrix (as a torch tensor) for the molecule.
    Each edge is weighted by the encoded bond type.
    """
    bond_encoder = bond_encoder or {}
    adjacency = torch.zeros((max_atoms, max_atoms))
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start < max_atoms and end < max_atoms:
            weight = bond_encoder.get(bond.GetBondType(), 0)
            adjacency[start, end] = weight
            adjacency[end, start] = weight
    return adjacency

def find_paths(G, u, n):
    """
    Recursively find all simple paths of length n from node u in graph G.
    """
    if n == 0:
        return [[u]]
    return [
        [u] + path 
        for neighbor in G.neighbors(u) 
        for path in find_paths(G, neighbor, n - 1) 
        if u not in path
    ]

def find_minimum_indices(arr):
    """
    Return the indices where the minimum value occurs in the list.
    """
    min_val = min(arr)
    return [index for index, value in enumerate(arr) if value == min_val]

def remove_unique_classes(class_matrix, feature_matrix):
    """
    Remove rows from class_matrix and feature_matrix where the class is unique.
    """
    unique_classes, class_counts = np.unique(class_matrix, axis=0, return_counts=True)
    unique_indices = np.where(class_counts == 1)[0]
    return np.delete(class_matrix, unique_indices, axis=0), np.delete(feature_matrix, unique_indices, axis=0)

def map_all_edges_g_to_G(g, G):
    """
    Map DGL graph edges to the original NetworkX graph node names.
    """
    edge_to_str_name_mapping = {}
    for edge_id in range(g.number_of_edges()):
        src_id, dst_id = g.find_edges(edge_id)
        src_name = g.ndata['name'][src_id].item()
        dst_name = g.ndata['name'][dst_id].item()
        src_in_G, dst_in_G = None, None
        for node, data in G.nodes(data=True):
            if data.get('name') == src_name:
                src_in_G = node
            if data.get('name') == dst_name:
                dst_in_G = node
            if src_in_G is not None and dst_in_G is not None:
                break
        if src_in_G is not None and dst_in_G is not None:
            edge_to_str_name_mapping[edge_id] = (src_in_G, dst_in_G)
    return edge_to_str_name_mapping
