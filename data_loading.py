# data_loading.py
import torch
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR
from sklearn.impute import SimpleImputer
from utils import get_hard_soft_value, feature_size, graph_adjacency

def DataExtN():
    """
    Loads polymer data from Excel files, constructs the graph (using NetworkX and DGL),
    and returns the DGL graph, the original NetworkX graph, a reverse mapping for nodes, 
    and a normalization constant.
    """
    import dgl
    # Load dataframes (ensure these files exist)
    df = pd.read_excel('database.xlsx', 'BCPs')
    dfH = pd.read_excel('homopolymers_overall_dp.xlsx')

    print('- Total BCP data:', df.shape[0])
    print('- Total Homopolymer data:', dfH.shape[0])
    
    imputer = SimpleImputer(missing_values=pd.NA, strategy='mean')
    # Extract features; adjust as needed for your project
    X = [df['σbreak (MPa)'], df['Lower Tg'], df['Upper Tg']]
    XE = [df['σbreak (MPa)']]
    XE1 = [df['εbreak (%)']]

    # Initialize variables
    wb = 1
    smiles = df['Big_Smile'].tolist()
    counts = df['Block DP']
    features = []
    features_weight = []
    features_weight_break = []
    G_nx = nx.DiGraph()
    Count = 0

    # Process homopolymer data
    HrdSoft = dfH['Hard/Soft'].tolist()

    print("Processing", len(smiles), "smiles...")
    for i in range(len(smiles)):
        # Parse the SMILES string (expected format: "{smile1}{smile2}...")  
        Sp = smiles[i][1:-1].split('}{')
        # Use first component for weight extraction
        features.append([XE[0][i] / wb])
        features_weight.append([XE[0][i] / wb])
        features_weight_break.append([XE1[0][i] / wb])

    # Build graph nodes and edges
    for i in range(len(smiles)):
        Sp = smiles[i][1:-1].split('}{')
        # Process Block DP counts
        Cn = str(counts[i]).split(':')
        if len(Cn[0].split('/')) == 1:
            Cn[0] = str(2 * float(Cn[0]))
        else:
            AC = Cn[0].split('/')
            Cn[0] = str(2 * float(AC[0])) + '/' + str(2 * float(AC[1]))

        # Keep only first two components for processing
        Sp = [Sp[0], Sp[1]]
        Spp = [s for s in Sp if len(s.split(',')) == 1]

        # Convert SMILES to molecules (add error check for invalid SMILES)
        data = [Chem.MolFromSmiles(line) for line in Spp if Chem.MolFromSmiles(line) is not None]
        if not data:
            continue  # Skip if no valid molecule

        atom_labels = sorted(set([atom.GetAtomicNum() for mol in data for atom in mol.GetAtoms()] + [0]))
        bond_labels = [Chem.rdchem.BondType.ZERO] + sorted({bond.GetBondType() for mol in data for bond in mol.GetBonds()})
        bond_encoder_m = {l: ii for ii, l in enumerate(bond_labels)}

        # Process features for each molecule
        features_list = []
        for idx, mol in enumerate(data):
            node_feature = torch.cat([
                feature_size(mol, atom_labels),
                graph_adjacency(mol, bond_encoder=bond_encoder_m)
            ], dim=1)
            # Weight the features by the square root of the DP value (first DP value)
            dp_value = torch.sqrt(torch.tensor(float(Cn[idx].split('/')[0]), dtype=torch.float))
            weighted_feature = (node_feature + 1) * dp_value
            features_list.append(weighted_feature.float())

        # Stack and pad features for consistency
        features_stack = torch.stack(features_list)
        features_stack = F.pad(features_stack, (0, 65 - features_stack.size(-1)), "constant", 0)

        # If more than one component exists, combine additional features
        if len(Sp[0].split(',')) > 1 or len(Sp[1].split(',')) > 1:
            # Process first additional component
            ss = Sp[0].split(',')
            jj = Cn[0].split('/')
            data1 = [Chem.MolFromSmiles(ss[0])]
            if data1[0] is not None:
                atom_labels1 = sorted(set([atom.GetAtomicNum() for mol in data1 for atom in mol.GetAtoms()] + [0]))
                bond_labels1 = [Chem.rdchem.BondType.ZERO] + sorted({bond.GetBondType() for mol in data1 for bond in mol.GetBonds()})
                bond_encoder1 = {l: ii for ii, l in enumerate(bond_labels1)}
                feat1 = []
                for mol in data1:
                    f = torch.cat([feature_size(mol, atom_labels1), graph_adjacency(mol, bond_encoder=bond_encoder1)], dim=1)
                    dp_w = torch.sqrt(torch.tensor(float(jj[0]), dtype=torch.float))
                    feat1.append((f + 1) * dp_w)
                feat1 = torch.stack(feat1)
                feat1 = F.pad(feat1, (0, 65 - feat1.size(-1)), "constant", 0)
            else:
                feat1 = features_stack[0:1]

            # Process second additional component
            ss2 = Sp[1].split(',')
            jj2 = Cn[1].split('/')
            data2 = [Chem.MolFromSmiles(ss2[0])]
            if data2[0] is not None:
                atom_labels2 = sorted(set([atom.GetAtomicNum() for mol in data2 for atom in mol.GetAtoms()] + [0]))
                bond_labels2 = [Chem.rdchem.BondType.ZERO] + sorted({bond.GetBondType() for mol in data2 for bond in mol.GetBonds()})
                bond_encoder2 = {l: ii for ii, l in enumerate(bond_labels2)}
                feat2 = []
                for mol in data2:
                    f = torch.cat([feature_size(mol, atom_labels2), graph_adjacency(mol, bond_encoder=bond_encoder2)], dim=1)
                    dp_w = torch.sqrt(torch.tensor(float(jj2[0]), dtype=torch.float))
                    feat2.append((f + 1) * dp_w)
                feat2 = torch.stack(feat2)
                feat2 = F.pad(feat2, (0, 65 - feat2.size(-1)), "constant", 0)
            else:
                feat2 = features_stack[0:1]

            # Combine additional features
            combined_feat = feat1 + feat2
            features_stack = torch.cat((features_stack, combined_feat), dim=0)

        # Pad each feature vector to a fixed length (e.g. 4000)
        padded_features = []
        for feat in features_stack:
            feat_flat = feat.view(-1)
            padding_length = 4000 - feat_flat.shape[0]
            padded_features.append(torch.cat([feat_flat, torch.zeros(padding_length)]))
        padded_features = torch.stack(padded_features)

        # Convert extra features into tensors
        feat_weight = torch.tensor(features[i])
        feat_weight_break = torch.tensor(features_weight_break[i])

        # Get hard/soft values for nodes (using first two smiles components)
        value0 = get_hard_soft_value(Sp[0], dfH)
        value1 = get_hard_soft_value(Sp[1], dfH)

        # Add nodes with appropriate bipartite attribute based on hard/soft values
        node_labels = []
        if value0 is not None and value1 is None:
            node_labels = [(Sp[0] + ';' + Cn[0], value0), (Sp[1] + ';' + Cn[1], 1 - value0)]
        elif value1 is not None and value0 is None:
            node_labels = [(Sp[0] + ';' + Cn[0], 1 - value1), (Sp[1] + ';' + Cn[1], value1)]
        elif value1 is not None and value0 is not None:
            node_labels = [(Sp[0] + ';' + Cn[0], value0), (Sp[1] + ';' + Cn[1], value1)]
        else:
            node_labels = [(Sp[0] + ';' + Cn[0], 0), (Sp[1] + ';' + Cn[1], 1)]

        for node_label, bip_val in node_labels:
            if not G_nx.has_node(node_label):
                Count += 1
                G_nx.add_node(node_label, Feature=padded_features[0], weight=feat_weight,
                              bipartite=bip_val, weightBreak=feat_weight_break, name=Count)

        # Add edge between the two nodes
        G_nx.add_edge(Sp[0] + ';' + Cn[0], Sp[1] + ';' + Cn[1],
                      Feature=padded_features[0], weight=feat_weight, weightBreak=feat_weight_break)

    # Create a mapping for nodes and convert NetworkX graph to DGL graph
    mapping = {node: i for i, node in enumerate(G_nx.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    import dgl
    g_dgl = dgl.from_networkx(G_nx, node_attrs=['Feature', 'weight', 'weightBreak', 'bipartite', 'name'],
                              edge_attrs=['Feature', 'weight', 'weightBreak'])
    
    return g_dgl, G_nx, reverse_mapping, wb

