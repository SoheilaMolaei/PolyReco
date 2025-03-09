import random
import numpy as np
import torch
import dgl
from sklearn.model_selection import train_test_split, KFold
import itertools

from data_extraction import DataExtN
from models import GraphSAGE, DotPredictor
from train_utils import init_weights, compute_loss, compute_auc, calculate_rmse
from helpers import map_all_edges_g_to_G

def main():
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the data and construct graphs
    g, G, reverse_mapping = DataExtN()
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())

    # Generate negative edges using bipartite node subset
    bipartite_nodes = np.where(g.ndata["bipartite"] == 1)[0]
    current_edges = set(zip(u.numpy(), v.numpy()))
    possible_neg_edges = [(x, y) for x in bipartite_nodes for y in bipartite_nodes if x != y]
    neg_edges = [edge for edge in possible_neg_edges if edge not in current_edges]
    if not neg_edges:
        raise ValueError("No negative edges found. Check your bipartite node definitions.")
    neg_u, neg_v = zip(*neg_edges)

    # Set up 2-fold cross-validation
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)

    for fold, (train_val_index, test_index) in enumerate(kf.split(eids)):
        train_index, val_index = train_test_split(train_val_index, test_size=0.5, random_state=seed)
        train_size = len(train_index)
        val_size = len(val_index)
        test_size = len(test_index)

        # Select positive edges for train/validation/test
        train_pos_u, train_pos_v = u[train_index], v[train_index]
        val_pos_u, val_pos_v = u[val_index], v[val_index]
        test_pos_u, test_pos_v = u[test_index], v[test_index]

        # Sample negative edges to match the number of positive edges
        neg_indices = np.random.permutation(len(neg_u))
        train_neg_u = np.array(neg_u)[neg_indices[:train_size]]
        train_neg_v = np.array(neg_v)[neg_indices[:train_size]]
        val_neg_u = np.array(neg_u)[neg_indices[train_size:train_size + val_size]]
        val_neg_v = np.array(neg_v)[neg_indices[train_size:train_size + val_size]]
        test_neg_u = np.array(neg_u)[neg_indices[train_size + val_size:train_size + val_size + test_size]]
        test_neg_v = np.array(neg_v)[neg_indices[train_size + val_size:train_size + val_size + test_size]]

        # Create training graph by removing a subset of edges (train+val) from the full graph
        train_g = dgl.remove_edges(g, eids[:train_size + val_size])

        # Helper to build subgraphs from edges
        def build_subgraph(pos_u, pos_v):
            sub_g = dgl.graph((pos_u, pos_v), num_nodes=g.number_of_nodes())
            sub_g.ndata.update(g.ndata)
            return sub_g

        train_pos_g = build_subgraph(train_pos_u, train_pos_v)
        val_pos_g   = build_subgraph(val_pos_u,   val_pos_v)
        test_pos_g  = build_subgraph(test_pos_u,  test_pos_v)

        # Negative-edge subgraphs
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
        train_neg_g.ndata.update(g.ndata)
        val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.number_of_nodes())
        val_neg_g.ndata.update(g.ndata)
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
        test_neg_g.ndata.update(g.ndata)

        # Initialize model and predictor
        in_feats = g.ndata['Feature'].shape[1]
        model = GraphSAGE(in_feats, 32)
        model.apply(init_weights)
        pred = DotPredictor()

        optimizer = torch.optim.Adam(
            itertools.chain(model.parameters(), pred.parameters()),
            lr=0.001, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

        best_val_loss = float('inf')
        patience = 1000
        no_improvement_count = 0

        # ---------------------------
        # Training / Validation Loop
        # ---------------------------
        for e in range(500):
            model.train()
            h = model(train_g, train_g.ndata['Feature'].float())
            pos_score, _ = pred(train_pos_g, h)
            neg_score, _ = pred(train_neg_g, h)
            loss = compute_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_h = model(train_g, train_g.ndata['Feature'].float())
                val_pos_score, _ = pred(val_pos_g, val_h)
                val_neg_score, _ = pred(val_neg_g, val_h)
                val_loss = compute_loss(val_pos_score, val_neg_score)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if e % 20 == 0:
                print(f'Epoch {e}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

            if no_improvement_count >= patience:
                print("Early stopping due to no improvement")
                break

        # --------------------
        # Test / Evaluation
        # --------------------
        model.eval()
        with torch.no_grad():
            test_h = model(test_pos_g, test_pos_g.ndata['Feature'].float())
            pos_score, _ = pred(test_pos_g, test_h)
            neg_score, _ = pred(test_neg_g, test_h)
            auc = compute_auc(pos_score, neg_score)
            print(f'Fold {fold + 1}, AUC: {auc:.4f}')

            # Predict on the full graph
            full_score, gF = pred(g, test_h)
            edge_to_str_name_mapping = map_all_edges_g_to_G(gF, G)

    return edge_to_str_name_mapping , gF
if __name__ == "__main__":
    main()