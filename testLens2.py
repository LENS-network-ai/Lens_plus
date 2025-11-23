#!/usr/bin/env python
"""
LENS2 Model Testing Script - Auto-Configuration Version

This script automatically detects ALL hyperparameters from checkpoint
and tests your LENS2 model with minimal configuration needed.

Features:
- Auto-detects architecture, regularization, and optimization parameters
- Bootstrap ROC/PR analysis with confidence intervals
- Edge sparsity analysis
- Publication-quality visualizations
- Works with both constrained and penalty optimization modes

Usage:
    python test_lens2_auto.py \
        --model-path best_model.pt \
        --test-data test.txt \
        --data-root /path/to/data \
        --output-dir test_results
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import json
from collections import defaultdict

# Import your modules
from utils.dataset import GraphDataset
from model.LENS2 import ImprovedEdgeGNN
from helper import Evaluator, collate, preparefeatureLabel


# ============================================================================
# CONFIGURATION EXTRACTION
# ============================================================================

def extract_config_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Extract ALL configuration from checkpoint
    
    Returns:
        config: Dictionary with all model parameters
    """
    print(f"\n{'='*70}")
    print(f"📋 EXTRACTING CONFIGURATION FROM CHECKPOINT")
    print(f"{'='*70}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Initialize config with defaults
    config = {
        # Basic info
        'checkpoint_path': checkpoint_path,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_accuracy': checkpoint.get('best_val_accuracy', 'unknown'),
        
        # Architecture
        'feature_dim': 512,
        'hidden_dim': 256,
        'num_classes': 3,
        'edge_dim': 128,
        'num_gnn_layers': 3,
        'num_attention_heads': 4,
        'use_attention_pooling': True,
        'dropout': 0.2,
        
        # Regularization
        'reg_mode': 'l0',
        'l0_method': 'hard-concrete',
        'lambda_reg': 0.01,
        'use_constrained': False,
        'constraint_target': 0.30,
        
        # L0 parameters
        'l0_gamma': -0.1,
        'l0_zeta': 1.1,
        'l0_beta': 0.66,
        
        # Schedule
        'warmup_epochs': 15,
        'initial_temp': 5.0,
        'graph_size_adaptation': False,
        'min_edges_per_node': 2.0,
    }
    
    # Extract from checkpoint if available
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        config.update(ckpt_config)
    
    # Extract from hyperparameters if available
    if 'hyperparameters' in checkpoint:
        hypers = checkpoint['hyperparameters']
        config.update(hypers)
    
    # Try to infer from state_dict
    state_dict = checkpoint.get('model_state_dict', {})
    
    # Infer num_classes from classifier
    for key in state_dict.keys():
        if 'classifier' in key and 'weight' in key:
            if state_dict[key].dim() >= 2:
                config['num_classes'] = state_dict[key].shape[0]
                break
    
    # Infer hidden_dim from GNN layers
    for key in state_dict.keys():
        if 'gnn_layers.0' in key and 'weight' in key:
            if state_dict[key].dim() >= 2:
                config['hidden_dim'] = state_dict[key].shape[0]
                break
    
    # Infer edge_dim from edge_scorer
    for key in state_dict.keys():
        if 'edge_scorer.edge_mlp.0' in key and 'weight' in key:
            if state_dict[key].dim() >= 2:
                config['edge_dim'] = state_dict[key].shape[0]
                break
    
    # Infer num_gnn_layers
    max_gnn_layer = 0
    for key in state_dict.keys():
        if 'gnn_layers.' in key:
            try:
                layer_num = int(key.split('gnn_layers.')[1].split('.')[0])
                max_gnn_layer = max(max_gnn_layer, layer_num)
            except:
                pass
    if max_gnn_layer > 0:
        config['num_gnn_layers'] = max_gnn_layer + 1
    
    # Print extracted configuration
    print(f"\n  Model Information:")
    print(f"  • Checkpoint Epoch: {config['epoch']}")
    print(f"  • Best Val Accuracy: {config.get('best_val_accuracy', 'N/A')}")
    
    print(f"\n  Architecture:")
    print(f"  • Feature Dim: {config['feature_dim']}")
    print(f"  • Hidden Dim: {config['hidden_dim']}")
    print(f"  • Edge Dim: {config['edge_dim']}")
    print(f"  • Num Classes: {config['num_classes']}")
    print(f"  • GNN Layers: {config['num_gnn_layers']}")
    print(f"  • Attention Heads: {config['num_attention_heads']}")
    print(f"  • Attention Pooling: {config['use_attention_pooling']}")
    print(f"  • Dropout: {config['dropout']}")
    
    print(f"\n  Regularization:")
    print(f"  • Mode: {config['reg_mode']}")
    print(f"  • L0 Method: {config['l0_method']}")
    print(f"  • Optimization: {'CONSTRAINED' if config['use_constrained'] else 'PENALTY'}")
    if config['use_constrained']:
        print(f"  • Constraint Target: {config['constraint_target']*100:.1f}%")
    else:
        print(f"  • Lambda Reg: {config['lambda_reg']}")
    
    print(f"{'='*70}\n")
    
    return config


# ============================================================================
# TOP-K EDGE SELECTION
# ============================================================================

def keep_top_k_edges(edge_weights, adj_matrix, ratio=0.30):
    """
    Keep only top K% edges by learned weight
    
    Args:
        edge_weights: Edge weights [B, N, N]
        adj_matrix: Adjacency matrix [B, N, N]
        ratio: Fraction of edges to keep
    
    Returns:
        Binary edge weights with top K% set to 1
    """
    edge_mask = (adj_matrix > 0).float()
    batch_size = edge_weights.shape[0]
    result = torch.zeros_like(edge_weights)
    
    for b in range(batch_size):
        mask_b = edge_mask[b] > 0
        weights_b = edge_weights[b][mask_b]
        
        if weights_b.numel() == 0:
            continue
        
        # Calculate number of edges to keep
        k = max(1, int(weights_b.numel() * ratio))
        
        # Get threshold for top k edges
        threshold = torch.topk(weights_b, k)[0][-1].item()
        
        # Keep edges >= threshold
        result[b] = (edge_weights[b] >= threshold).float() * edge_mask[b]
    
    return result


# ============================================================================
# BOOTSTRAP ANALYSIS
# ============================================================================

def bootstrap_roc_pr_analysis(y_true, y_score_probs, n_bootstrap=10000, confidence_level=0.95):
    """
    Compute bootstrap confidence intervals for ROC AUC, PR AUC, and curves
    """
    n_classes = y_score_probs.shape[1]
    n_samples = len(y_true)
    
    # Convert one-hot to class labels if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
    
    # Initialize storage
    roc_auc_scores = {i: [] for i in range(n_classes)}
    roc_auc_scores['macro'] = []
    roc_auc_scores['weighted'] = []
    
    pr_auc_scores = {i: [] for i in range(n_classes)}
    pr_auc_scores['macro'] = []
    pr_auc_scores['weighted'] = []
    
    fpr_grid = np.linspace(0, 1, 100)
    recall_grid = np.linspace(0, 1, 100)
    
    tpr_values = {i: np.zeros((n_bootstrap, len(fpr_grid))) for i in range(n_classes)}
    precision_values = {i: np.zeros((n_bootstrap, len(recall_grid))) for i in range(n_classes)}
    
    print(f"Running bootstrap with {n_bootstrap} iterations...")
    
    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        y_true_boot = y_true_labels[indices]
        y_score_boot = y_score_probs[indices]
        
        y_true_bin = label_binarize(y_true_boot, classes=range(n_classes))
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        for i in range(n_classes):
            try:
                roc_auc = roc_auc_score(y_true_bin[:, i], y_score_boot[:, i])
                roc_auc_scores[i].append(roc_auc)
                
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_boot[:, i])
                tpr_interp = np.interp(fpr_grid, fpr, tpr)
                tpr_values[i][b] = tpr_interp
                
                pr_auc = average_precision_score(y_true_bin[:, i], y_score_boot[:, i])
                pr_auc_scores[i].append(pr_auc)
                
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score_boot[:, i])
                precision_interp = np.interp(recall_grid[::-1], recall[::-1], precision[::-1])[::-1]
                precision_values[i][b] = precision_interp
                
            except ValueError:
                roc_auc_scores[i].append(np.nan)
                pr_auc_scores[i].append(np.nan)
                tpr_values[i][b] = np.nan
                precision_values[i][b] = np.nan
        
        valid_roc = [roc_auc_scores[i][-1] for i in range(n_classes) if not np.isnan(roc_auc_scores[i][-1])]
        valid_pr = [pr_auc_scores[i][-1] for i in range(n_classes) if not np.isnan(pr_auc_scores[i][-1])]
        
        if valid_roc:
            class_counts = np.bincount(y_true_boot, minlength=n_classes)
            
            roc_auc_scores['macro'].append(np.mean(valid_roc))
            roc_auc_scores['weighted'].append(np.average(valid_roc, weights=class_counts[:len(valid_roc)]))
            
            pr_auc_scores['macro'].append(np.mean(valid_pr))
            pr_auc_scores['weighted'].append(np.average(valid_pr, weights=class_counts[:len(valid_pr)]))
        else:
            roc_auc_scores['macro'].append(np.nan)
            roc_auc_scores['weighted'].append(np.nan)
            pr_auc_scores['macro'].append(np.nan)
            pr_auc_scores['weighted'].append(np.nan)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    results = {}
    
    for i in range(n_classes):
        valid_roc = [s for s in roc_auc_scores[i] if not np.isnan(s)]
        valid_pr = [s for s in pr_auc_scores[i] if not np.isnan(s)]
        valid_tprs = tpr_values[i][~np.isnan(tpr_values[i]).any(axis=1)]
        valid_precisions = precision_values[i][~np.isnan(precision_values[i]).any(axis=1)]
        
        if valid_roc and valid_pr:
            results[f'class_{i}'] = {
                'roc_auc_mean': np.mean(valid_roc),
                'roc_auc_std': np.std(valid_roc),
                'roc_auc_ci_low': np.percentile(valid_roc, 100 * alpha / 2),
                'roc_auc_ci_high': np.percentile(valid_roc, 100 * (1 - alpha / 2)),
                'tpr_mean': np.mean(valid_tprs, axis=0),
                'tpr_std': np.std(valid_tprs, axis=0),
                'tpr_ci_low': np.percentile(valid_tprs, 100 * alpha / 2, axis=0),
                'tpr_ci_high': np.percentile(valid_tprs, 100 * (1 - alpha / 2), axis=0),
                'fpr_grid': fpr_grid,
                'pr_auc_mean': np.mean(valid_pr),
                'pr_auc_std': np.std(valid_pr),
                'pr_auc_ci_low': np.percentile(valid_pr, 100 * alpha / 2),
                'pr_auc_ci_high': np.percentile(valid_pr, 100 * (1 - alpha / 2)),
                'precision_mean': np.mean(valid_precisions, axis=0),
                'precision_std': np.std(valid_precisions, axis=0),
                'precision_ci_low': np.percentile(valid_precisions, 100 * alpha / 2, axis=0),
                'precision_ci_high': np.percentile(valid_precisions, 100 * (1 - alpha / 2), axis=0),
                'recall_grid': recall_grid
            }
    
    for avg_type in ['macro', 'weighted']:
        valid_roc = [s for s in roc_auc_scores[avg_type] if not np.isnan(s)]
        valid_pr = [s for s in pr_auc_scores[avg_type] if not np.isnan(s)]
        
        if valid_roc and valid_pr:
            results[avg_type] = {
                'roc_auc_mean': np.mean(valid_roc),
                'roc_auc_std': np.std(valid_roc),
                'roc_auc_ci_low': np.percentile(valid_roc, 100 * alpha / 2),
                'roc_auc_ci_high': np.percentile(valid_roc, 100 * (1 - alpha / 2)),
                'pr_auc_mean': np.mean(valid_pr),
                'pr_auc_std': np.std(valid_pr),
                'pr_auc_ci_low': np.percentile(valid_pr, 100 * alpha / 2),
                'pr_auc_ci_high': np.percentile(valid_pr, 100 * (1 - alpha / 2))
            }
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_roc_pr_curves(results, n_classes, output_path, class_names=None):
    """Plot ROC and PR curves with confidence bands"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # ROC curves
    for i in range(n_classes):
        class_key = f'class_{i}'
        if class_key in results:
            res = results[class_key]
            
            ax1.plot(res['fpr_grid'], res['tpr_mean'], 
                    color=colors[i], linewidth=2,
                    label=f'{class_names[i]} (AUC={res["roc_auc_mean"]:.3f}±{res["roc_auc_std"]:.3f})')
            
            ax1.fill_between(res['fpr_grid'], 
                           res['tpr_ci_low'], res['tpr_ci_high'],
                           color=colors[i], alpha=0.2)
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title('ROC Curves with 95% CI', fontsize=16)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # PR curves
    for i in range(n_classes):
        class_key = f'class_{i}'
        if class_key in results:
            res = results[class_key]
            
            ax2.plot(res['recall_grid'], res['precision_mean'], 
                    color=colors[i], linewidth=2,
                    label=f'{class_names[i]} (AP={res["pr_auc_mean"]:.3f}±{res["pr_auc_std"]:.3f})')
            
            ax2.fill_between(res['recall_grid'], 
                           res['precision_ci_low'], res['precision_ci_high'],
                           color=colors[i], alpha=0.2)
    
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title('Precision-Recall Curves with 95% CI', fontsize=16)
    ax2.legend(loc="lower left", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_edge_sparsity(edge_densities, edge_weights_all, output_path):
    """Plot edge sparsity analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Edge weight distribution
    ax1.hist(edge_weights_all, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(edge_weights_all), color='r', linestyle='--', 
               label=f'Mean={np.mean(edge_weights_all):.3f}')
    ax1.set_xlabel('Edge Weight', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Edge Weight Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Density per sample
    ax2.hist(edge_densities, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(np.mean(edge_densities), color='r', linestyle='--', 
               label=f'Mean={np.mean(edge_densities):.3f}')
    ax2.set_xlabel('Edge Density', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Per-Sample Edge Density', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def calculate_metrics(y_true, y_pred_probs):
    """Calculate comprehensive classification metrics"""
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
    
    return {
        'accuracy': accuracy_score(y_true_labels, y_pred),
        'precision_macro': precision_score(y_true_labels, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true_labels, y_pred, average='weighted', zero_division=0),
        'precision_per_class': precision_score(y_true_labels, y_pred, average=None, zero_division=0),
        'recall_macro': recall_score(y_true_labels, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true_labels, y_pred, average='weighted', zero_division=0),
        'recall_per_class': recall_score(y_true_labels, y_pred, average=None, zero_division=0),
        'f1_macro': f1_score(y_true_labels, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true_labels, y_pred, average='weighted', zero_division=0),
        'f1_per_class': f1_score(y_true_labels, y_pred, average=None, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true_labels, y_pred),
        'classification_report': classification_report(y_true_labels, y_pred, output_dict=True, zero_division=0)
    }


def analyze_edge_sparsity(model, test_loader, device, config, output_dir, use_top_k=True, top_k_ratio=0.30):
    """Analyze edge sparsity with optional top-k selection"""
    model.eval()
    
    edge_densities = []
    edge_weights_all = []
    num_nodes_list = []
    
    print(f"\n📊 Analyzing edge sparsity...")
    if use_top_k:
        print(f"   Using TOP-{top_k_ratio*100:.0f}% edge selection")
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Processing"):
            node_feat, labels, adjs, masks = preparefeatureLabel(
                sample['image'], sample['label'], sample['adj_s'], 
                n_features=config['feature_dim']
            )
            
            if hasattr(model, 'edge_scorer'):
                edge_weights, _ = model.edge_scorer.compute_edge_weights(
                    node_feat=node_feat,
                    adj_matrix=adjs,
                    current_epoch=999,
                    warmup_epochs=0,
                    temperature=config['initial_temp'],
                    regularizer=model.regularizer if hasattr(model, 'regularizer') else None,
                    use_l0=True,
                    training=False
                )
                
                # Apply top-k if requested
                if use_top_k:
                    edge_weights = keep_top_k_edges(edge_weights, adjs, ratio=top_k_ratio)
                
                # Calculate density
                edge_mask = (adjs > 0).float()
                num_edges = edge_mask.sum().item()
                active_edges = (edge_weights * edge_mask).sum().item()
                density = active_edges / (num_edges + 1e-8)
                
                edge_densities.append(density)
                
                valid_weights = edge_weights[edge_mask > 0].cpu().numpy()
                edge_weights_all.extend(valid_weights)
                
                num_nodes_list.append(node_feat.shape[1])
    
    edge_weights_all = np.array(edge_weights_all)
    
    stats = {
        'mean_density': np.mean(edge_densities),
        'std_density': np.std(edge_densities),
        'min_density': np.min(edge_densities),
        'max_density': np.max(edge_densities),
        'mean_edge_weight': np.mean(edge_weights_all),
        'median_edge_weight': np.median(edge_weights_all),
        'std_edge_weight': np.std(edge_weights_all),
        'sparsity_threshold_01': (edge_weights_all > 0.1).mean() * 100,
        'sparsity_threshold_05': (edge_weights_all > 0.5).mean() * 100,
        'sparsity_threshold_09': (edge_weights_all > 0.9).mean() * 100,
        'avg_num_nodes': np.mean(num_nodes_list),
    }
    
    # Plot
    plot_edge_sparsity(edge_densities, edge_weights_all, 
                      os.path.join(output_dir, 'edge_sparsity_analysis.png'))
    
    return stats


# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def test_lens2_model(args):
    """Main testing function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🧪 LENS2 MODEL TESTING - AUTO-CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Test Data: {args.test_data}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Extract configuration from checkpoint
    config = extract_config_from_checkpoint(args.model_path, device)
    
    # Load test data
    print("📂 Loading test dataset...")
    with open(args.test_data, 'r') as f:
        test_ids = f.readlines()
    
    test_dataset = GraphDataset(root=args.data_root, ids=test_ids)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)
    
    print(f"   ✓ Loaded {len(test_dataset)} test samples\n")
    
    # Create model
    print("🔧 Loading model...")
    model = ImprovedEdgeGNN(
        feature_dim=config['feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        num_gnn_layers=config['num_gnn_layers'],
        num_attention_heads=config['num_attention_heads'],
        use_attention_pooling=config['use_attention_pooling'],
        lambda_reg=config['lambda_reg'],
        reg_mode=config['reg_mode'],
        l0_method=config['l0_method'],
        use_constrained=config['use_constrained'],
        constraint_target=config['constraint_target'],
        edge_dim=config['edge_dim'],
        dropout=config['dropout'],
        warmup_epochs=config['warmup_epochs'],
        graph_size_adaptation=config['graph_size_adaptation'],
        min_edges_per_node=config['min_edges_per_node'],
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✓ Model loaded successfully\n")
    
    # Collect predictions
    evaluator = Evaluator(n_class=config['num_classes'])
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("🔍 Collecting predictions...")
    if args.use_top_k:
        print(f"   Using TOP-{args.top_k_ratio*100:.0f}% edge selection")
    
    # Monkey-patch for top-k if enabled
    if args.use_top_k:
        original_compute = model.edge_scorer.compute_edge_weights
        
        def compute_with_topk(*pargs, **kwargs):
            edge_weights, logAlpha = original_compute(*pargs, **kwargs)
            if 'adj_matrix' in kwargs:
                edge_weights = keep_top_k_edges(edge_weights, kwargs['adj_matrix'], ratio=args.top_k_ratio)
            elif len(pargs) >= 2:
                edge_weights = keep_top_k_edges(edge_weights, pargs[1], ratio=args.top_k_ratio)
            return edge_weights, logAlpha
        
        model.edge_scorer.compute_edge_weights = compute_with_topk
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Processing"):
            pred, labels, _, _ = evaluator.eval_test(sample, model, n_features=config['feature_dim'])
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(pred.cpu().numpy())
    
    # Restore original method if patched
    if args.use_top_k:
        model.edge_scorer.compute_edge_weights = original_compute
    
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    print(f"   ✓ Collected {len(all_labels)} predictions\n")
    
    # Calculate metrics
    print("📊 Calculating metrics...")
    metrics = calculate_metrics(all_labels, all_probabilities)
    
    print(f"\n   Overall Performance:")
    print(f"   • Accuracy: {metrics['accuracy']:.4f}")
    print(f"   • Macro F1: {metrics['f1_macro']:.4f}")
    print(f"   • Weighted F1: {metrics['f1_weighted']:.4f}\n")
    
    # Bootstrap analysis
    if args.n_bootstrap > 0:
        print(f"🔄 Running bootstrap analysis ({args.n_bootstrap} iterations)...")
        bootstrap_results = bootstrap_roc_pr_analysis(
            all_labels, 
            all_probabilities, 
            n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level
        )
    else:
        bootstrap_results = None
        print("⏭️  Skipping bootstrap analysis\n")
    
    # Edge sparsity analysis
    if args.analyze_sparsity:
        sparsity_stats = analyze_edge_sparsity(
            model, test_loader, device, config, args.output_dir,
            use_top_k=args.use_top_k, top_k_ratio=args.top_k_ratio
        )
    else:
        sparsity_stats = None
        print("⏭️  Skipping sparsity analysis\n")
    
    # Generate plots
    print("📈 Generating visualizations...")
    
    class_names = args.class_names.split(',') if args.class_names else [f'Class {i}' for i in range(config['num_classes'])]
    
    if bootstrap_results:
        plot_roc_pr_curves(
            bootstrap_results, 
            config['num_classes'], 
            os.path.join(args.output_dir, 'roc_pr_curves.png'),
            class_names=class_names
        )
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Save results
    print("\n💾 Saving results...")
    
    results_file = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LENS2 MODEL TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("Model Information:\n")
        f.write("-"*40 + "\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Checkpoint Epoch: {config['epoch']}\n")
        f.write(f"Best Val Accuracy: {config.get('best_val_accuracy', 'N/A')}\n")
        f.write(f"Test Data: {args.test_data}\n")
        f.write(f"Number of Samples: {len(all_labels)}\n")
        f.write(f"Number of Classes: {config['num_classes']}\n")
        f.write("\n")
        
        f.write("Model Configuration:\n")
        f.write("-"*40 + "\n")
        for key, value in config.items():
            if key not in ['checkpoint_path']:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Overall Metrics:\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1: {metrics['f1_weighted']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"Weighted Precision: {metrics['precision_weighted']:.4f}\n")
        f.write(f"Macro Recall: {metrics['recall_macro']:.4f}\n")
        f.write(f"Weighted Recall: {metrics['recall_weighted']:.4f}\n")
        f.write("\n")
        
        if bootstrap_results:
            f.write("ROC/PR AUC with Bootstrap CI:\n")
            f.write("-"*40 + "\n")
            
            for i in range(config['num_classes']):
                class_key = f'class_{i}'
                if class_key in bootstrap_results:
                    res = bootstrap_results[class_key]
                    f.write(f"\n{class_names[i]}:\n")
                    f.write(f"  ROC AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f}\n")
                    f.write(f"  ROC CI: [{res['roc_auc_ci_low']:.4f}, {res['roc_auc_ci_high']:.4f}]\n")
                    f.write(f"  PR AUC: {res['pr_auc_mean']:.4f} ± {res['pr_auc_std']:.4f}\n")
                    f.write(f"  PR CI: [{res['pr_auc_ci_low']:.4f}, {res['pr_auc_ci_high']:.4f}]\n")
                    f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                    f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                    f.write(f"  F1: {metrics['f1_per_class'][i]:.4f}\n")
            
            if 'macro' in bootstrap_results:
                res = bootstrap_results['macro']
                f.write("\nMacro Average:\n")
                f.write(f"  ROC AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f}\n")
                f.write(f"  ROC CI: [{res['roc_auc_ci_low']:.4f}, {res['roc_auc_ci_high']:.4f}]\n")
                f.write(f"  PR AUC: {res['pr_auc_mean']:.4f} ± {res['pr_auc_std']:.4f}\n")
                f.write(f"  PR CI: [{res['pr_auc_ci_low']:.4f}, {res['pr_auc_ci_high']:.4f}]\n")
        
        if sparsity_stats:
            f.write("\nEdge Sparsity Analysis:\n")
            f.write("-"*40 + "\n")
            f.write(f"Mean Density: {sparsity_stats['mean_density']:.4f} ± {sparsity_stats['std_density']:.4f}\n")
            f.write(f"Density Range: [{sparsity_stats['min_density']:.4f}, {sparsity_stats['max_density']:.4f}]\n")
            f.write(f"Mean Edge Weight: {sparsity_stats['mean_edge_weight']:.4f}\n")
            f.write(f"Std Edge Weight: {sparsity_stats['std_edge_weight']:.4f}\n")
            f.write(f"Median Edge Weight: {sparsity_stats['median_edge_weight']:.4f}\n")
            f.write(f"Edges > 0.1: {sparsity_stats['sparsity_threshold_01']:.2f}%\n")
            f.write(f"Edges > 0.5: {sparsity_stats['sparsity_threshold_05']:.2f}%\n")
            f.write(f"Edges > 0.9: {sparsity_stats['sparsity_threshold_09']:.2f}%\n")
            f.write(f"Avg Nodes per Graph: {sparsity_stats['avg_num_nodes']:.1f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write("-"*40 + "\n")
        f.write(str(metrics['confusion_matrix']) + "\n")
    
    # Save JSON
    json_results = {
        'model_info': {
            'checkpoint_path': args.model_path,
            'epoch': config['epoch'],
            'best_val_accuracy': config.get('best_val_accuracy', None),
        },
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
        },
        'config': {k: v for k, v in config.items() if k != 'checkpoint_path'},
    }
    
    if bootstrap_results and 'macro' in bootstrap_results:
        json_results['metrics']['roc_auc_macro'] = float(bootstrap_results['macro']['roc_auc_mean'])
        json_results['metrics']['pr_auc_macro'] = float(bootstrap_results['macro']['pr_auc_mean'])
    
    if sparsity_stats:
        json_results['sparsity'] = {k: float(v) for k, v in sparsity_stats.items()}
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_dir}")
    print(f"   • test_results.txt")
    print(f"   • results.json")
    print(f"   • roc_pr_curves.png")
    print(f"   • confusion_matrix.png")
    if sparsity_stats:
        print(f"   • edge_sparsity_analysis.png")
    
    print(f"\n{'='*70}")
    print(f"✅ TESTING COMPLETE")
    print(f"{'='*70}\n")
    
    return metrics, bootstrap_results, sparsity_stats, config


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test LENS2 Model - Auto-Configuration')
    
    # Required
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True, 
                       help='Path to test data file (list of IDs)')
    parser.add_argument('--data-root', type=str, required=True, 
                       help='Root directory for dataset')
    
    # Testing parameters
    parser.add_argument('--n-bootstrap', type=int, default=10000, 
                       help='Bootstrap iterations (0 to skip)')
    parser.add_argument('--confidence-level', type=float, default=0.95, 
                       help='Confidence level')
    parser.add_argument('--analyze-sparsity', action='store_true', 
                       help='Analyze edge sparsity')
    parser.add_argument('--use-top-k', action='store_true',
                       help='Use top-k edge selection during testing')
    parser.add_argument('--top-k-ratio', type=float, default=0.30,
                       help='Ratio of top edges to keep (default: 0.30)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='test_results', 
                       help='Output directory')
    parser.add_argument('--class-names', type=str, default=None, 
                       help='Comma-separated class names')
    
    args = parser.parse_args()
    
    # Run testing
    metrics, bootstrap_results, sparsity_stats, config = test_lens2_model(args)
