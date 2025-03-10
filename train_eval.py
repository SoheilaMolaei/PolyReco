# train_eval.py
import os
import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from data_loading import DataExtN
from models import GraphSAGE, MLPPredictor, GraphDecoder, init_weights
from utils import map_all_edges_g_to_G

# Set seeds for reproducibility
seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compute_loss(pos_score, neg_score, weight_pos=1.0, weight_neg=1.0):
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score),
                                                    weight=torch.full_like(pos_score, weight_pos))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score),
                                                    weight=torch.full_like(neg_score, weight_neg))
    return pos_loss + neg_loss

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().cpu().numpy()
    return roc_auc_score(labels, scores)

def calculate_rmse(pred_scores, true_weights):
    pred_np = pred_scores.detach().cpu().numpy()
    true_np = true_weights.detach().cpu().numpy()
    mse = np.mean((pred_np - true_np) ** 2)
    return math.sqrt(mse)

def train_and_evaluate():
    # Load data and build graph
    g, G_nx, reverse_mapping, wb = DataExtN()
    u, v = g.edges()
    
    # Use KFold on edge IDs (customize splits as needed)
    eids = np.arange(g.number_of_edges())
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    bipartite_nodes = np.where(g.ndata["bipartite"] == 1)[0]
    current_edges = set(zip(u.numpy(), v.numpy()))
    possible_neg_edges = [(x, y) for x in bipartite_nodes for y in bipartite_nodes if x != y]
    neg_edges = [edge for edge in possible_neg_edges if edge not in current_edges]
    neg_u, neg_v = zip(*neg_edges)

    # For simplicity, use only one fold for training in this example
    for fold, (train_val_index, test_index) in enumerate(kf.split(eids)):
        train_index, val_index = train_test_split(train_val_index, test_size=0.5, random_state=seed)
        train_size, val_size, test_size = len(train_index), len(val_index), len(test_index)

        train_pos_u, train_pos_v = u[train_index], v[train_index]
        val_pos_u, val_pos_v = u[val_index], v[val_index]
        test_pos_u, test_pos_v = u[test_index], v[test_index]

        neg_indices = np.random.permutation(len(neg_u))
        train_neg_u = np.array(neg_u)[neg_indices[:train_size]]
        train_neg_v = np.array(neg_v)[neg_indices[:train_size]]
        val_neg_u = np.array(neg_u)[neg_indices[train_size:train_size + val_size]]
        val_neg_v = np.array(neg_v)[neg_indices[train_size:train_size + val_size]]
        test_neg_u = np.array(neg_u)[neg_indices[train_size + val_size:]]
        test_neg_v = np.array(neg_v)[neg_indices[train_size + val_size:]]

        # Construct subgraphs for training, validation, and testing
        train_g = dgl.remove_edges(g, eids[:train_size + val_size])
        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
        train_pos_g.ndata.update(g.ndata)
        for edge in zip(train_pos_u, train_pos_v):
            train_pos_g.edges[edge].data['weight'] = g.edges[edge].data['weight']
            train_pos_g.edges[edge].data['weightBreak'] = g.edges[edge].data['weightBreak']

        val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.number_of_nodes())
        val_pos_g.ndata.update(g.ndata)
        for edge in zip(val_pos_u, val_pos_v):
            val_pos_g.edges[edge].data['weight'] = g.edges[edge].data['weight']
            val_pos_g.edges[edge].data['weightBreak'] = g.edges[edge].data['weightBreak']

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
        test_pos_g.ndata.update(g.ndata)
        for edge in zip(test_pos_u, test_pos_v):
            test_pos_g.edges[edge].data['weight'] = g.edges[edge].data['weight']
            test_pos_g.edges[edge].data['weightBreak'] = g.edges[edge].data['weightBreak']

        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
        train_neg_g.ndata.update(g.ndata)
        val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.number_of_nodes())
        val_neg_g.ndata.update(g.ndata)
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
        test_neg_g.ndata.update(g.ndata)

        # Define model and predictor
        h_feats = 32
        model = GraphSAGE(g.ndata['Feature'].shape[1], h_feats)
        model.apply(init_weights)
        predictor = MLPPredictor(h_feats)

        #############################
        # Pre-training Stage
        #############################
        pretrain_epochs = 20000  # Adjust the number of pre-training epochs as needed
        decoder = GraphDecoder(h_feats, g.ndata['Feature'].shape[1])
        pretrain_optimizer = torch.optim.Adam(
            list(model.parameters()) + list(decoder.parameters()),
            lr=0.001, weight_decay=1e-4
        )
        mse_loss = nn.MSELoss()
        best_pretrain_loss = float('inf')
        print("Starting pre-training stage...")
        for epoch in range(pretrain_epochs):
            model.train()
            decoder.train()
            pretrain_optimizer.zero_grad()
            # Forward pass: encode and then decode to reconstruct the original features
            h_pre = model(train_g, train_g.ndata['Feature'].float())
            reconstructed = decoder(h_pre)
            loss_pretrain = mse_loss(reconstructed, train_g.ndata['Feature'].float())
            loss_pretrain.backward()
            pretrain_optimizer.step()
            if loss_pretrain < best_pretrain_loss:
                best_pretrain_loss = loss_pretrain.item()
            if epoch % 50 == 0:
                print(f"Pre-training Epoch {epoch}: Loss = {loss_pretrain.item():.4f}")
        print("Pre-training completed. Best pre-training loss:", best_pretrain_loss)

        #############################
        # Fine-tuning Stage (Link Prediction)
        #############################
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=0.001, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        max_patience = 1000

        # Training loop for link prediction
        for epoch in range(3000):
            model.train()
            h = model(train_g, train_g.ndata['Feature'].float())
            pos_score, _ = predictor(train_pos_g, h)
            neg_score, _ = predictor(train_neg_g, h)
            loss = compute_loss(pos_score, neg_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_h = model(train_g, train_g.ndata['Feature'].float())
                val_pos_score, _ = predictor(val_pos_g, val_h)
                val_neg_score, _ = predictor(val_neg_g, val_h)
                val_loss = compute_loss(val_pos_score, val_neg_score)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')
                torch.save(optimizer.state_dict(), 'best_optimizer.pth')
                torch.save(predictor.state_dict(), 'best_predictor.pth')
                patience_counter = 0
                print(f"Epoch {epoch}: Saved best model with val loss {val_loss.item():.4f}")
            else:
                patience_counter += 1

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

            if patience_counter >= max_patience:
                print("Early stopping triggered")
                break

        # Load best model for testing
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()

        # Testing phase
        with torch.no_grad():
            test_h = model(test_pos_g, test_pos_g.ndata['Feature'].float())
            pos_score, _ = predictor(test_pos_g, test_h)
            neg_score, _ = predictor(test_neg_g, test_h)
            auc = compute_auc(pos_score, neg_score)
            print(f"Test AUC: {auc:.4f}")

            # Predict on the full graph
            g_full = g.clone()
            full_h = model(g_full, g_full.ndata['Feature'].float())
            g_full.edata['score'], _ = predictor(g_full, full_h)

            # Map edges back to the original graph
            edge_mapping = map_all_edges_g_to_G(g_full, G_nx)

            # Save predictions and compute RMSE
            normalization_factor = 6  # adjust as necessary
            predicted_scores = []
            true_weights = []
            for edge_id, (u_name, v_name) in edge_mapping.items():
                if edge_id >= g_full.number_of_edges():
                    continue
                pred_score = g_full.edata['score'][edge_id].item() / normalization_factor
                true_weight = g_full.edata['weight'][edge_id].item()
                predicted_scores.append(pred_score)
                true_weights.append(true_weight)

            # Compute classification accuracy based on a threshold and tolerance
            threshold = 15.0
            tolerance = 0.0
            amb_lower = threshold - tolerance
            amb_upper = threshold + tolerance
            correct = 0
            for pred, true in zip(predicted_scores, true_weights):
                true_class = 1 if true > threshold else 0
                pred_class = 1 if pred > threshold else 0
                if amb_lower <= pred <= amb_upper:
                    correct += 1
                else:
                    if pred_class == true_class:
                        correct += 1
            accuracy = correct / len(true_weights)
            print(f"Classification Accuracy : {accuracy:.4f}")
        break  # Remove this if you want to run through all folds

