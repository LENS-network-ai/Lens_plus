#!/usr/bin/env python
"""
Comprehensive Testing Script for LENS2 Model

Features:
- Bootstrap ROC/PR analysis with confidence intervals
- Support for constrained and penalty optimization modes
- Multi-layer GNN and attention pooling support
- Detailed performance metrics and visualizations
- Edge sparsity analysis

Usage:
    python test_lens2.py \
        --model-path best_model.pt \
        --test-data test.txt \
        --data-root /path/to/data \
        --output-dir test_results
"""

import os
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
from scipy import stats
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import json

# Import your modules
from utils.dataset import GraphDataset
from model.LENS2 import ImprovedEdgeGNN  # Updated import
from helper import Evaluator, collate, preparefeatureLabel


def bootstrap_roc_pr_analysis(y_true, y_score_probs, n_bootstrap=10000, confidence_level=0.95):
    """
    Compute bootstrap confidence intervals for ROC AUC, PR AUC, and curves
    
    Args:
        y_true: True labels (one-hot encoded for multiclass)
        y_score_probs: Predicted probabilities for each class
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
    
    Returns:
        Dictionary containing ROC and PR statistics with confidence intervals
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
    
    # Fixed grids for interpolation
    fpr_grid = np.linspace(0, 1, 100)
    recall_grid = np.linspace(0, 1, 100)
    
    tpr_values = {i: np.zeros((n_bootstrap, len(fpr_grid))) for i in range(n_classes)}
    precision_values = {i: np.zeros((n_bootstrap, len(recall_grid))) for i in range(n_classes)}
    
    print(f"Running bootstrap with {n_bootstrap} iterations...")
    
    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        y_true_boot = y_true_labels[indices]
        y_score_boot = y_score_probs[indices]
        
        # Binarize labels
        y_true_bin = label_binarize(y_true_boot, classes=range(n_classes))
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        # Calculate per-class metrics
        for i in range(n_classes):
            try:
                # ROC
                roc_auc = roc_auc_score(y_true_bin[:, i], y_score_boot[:, i])
                roc_auc_scores[i].append(roc_auc)
                
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_boot[:, i])
                tpr_interp = np.interp(fpr_grid, fpr, tpr)
                tpr_values[i][b] = tpr_interp
                
                # PR
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
        
        # Macro and weighted averages
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


def plot_combined_curves(results, n_classes, output_path, class_names=None):
    """Plot combined ROC and PR curves with confidence bands"""
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


def analyze_edge_sparsity(model, test_loader, device, n_features, output_dir):
    """
    Analyze edge sparsity patterns across test samples
    
    Returns:
        dict: Sparsity statistics
    """
    model.eval()
    
    edge_densities = []
    edge_weights_all = []
    num_nodes_list = []
    
    print("\n📊 Analyzing edge sparsity...")
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Processing"):
            node_feat, labels, adjs, masks = preparefeatureLabel(
                sample['image'], sample['label'], sample['adj_s'], 
                n_features=n_features
            )
            
            # Get edge weights from model
            if hasattr(model, 'edge_scorer'):
                edge_weights, _ = model.edge_scorer.compute_edge_weights(
                    node_feat=node_feat,
                    adj_matrix=adjs,
                    current_epoch=999,  # Use final trained state
                    warmup_epochs=0,
                    temperature=1.0,
                    regularizer=model.regularizer if hasattr(model, 'regularizer') else None,
                    use_l0=hasattr(model, 'use_l0') and model.use_l0,
                    training=True
                )
                
                # Calculate density
                edge_mask = (adjs > 0).float()
                num_edges = edge_mask.sum().item()
                active_edges = (edge_weights * edge_mask).sum().item()
                density = active_edges / (num_edges + 1e-8)
                
                edge_densities.append(density)
                
                # Collect edge weights
                valid_weights = edge_weights[edge_mask > 0].cpu().numpy()
                edge_weights_all.extend(valid_weights)
                
                num_nodes_list.append(node_feat.shape[1])
    
    # Calculate statistics
    edge_weights_all = np.array(edge_weights_all)
    
    stats = {
        'mean_density': np.mean(edge_densities),
        'std_density': np.std(edge_densities),
        'min_density': np.min(edge_densities),
        'max_density': np.max(edge_densities),
        'mean_edge_weight': np.mean(edge_weights_all),
        'median_edge_weight': np.median(edge_weights_all),
        'sparsity_01': (edge_weights_all > 0.1).mean() * 100,
        'sparsity_05': (edge_weights_all > 0.5).mean() * 100,
        'sparsity_09': (edge_weights_all > 0.9).mean() * 100,
        'avg_num_nodes': np.mean(num_nodes_list),
    }
    
    # Plot edge weight distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax1.hist(edge_weights_all, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(0.1, color='r', linestyle='--', label='Threshold 0.1')
    ax1.axvline(0.5, color='g', linestyle='--', label='Threshold 0.5')
    ax1.set_xlabel('Edge Weight', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Edge Weight Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Density per sample
    ax2.hist(edge_densities, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(edge_densities), color='r', linestyle='--', 
               label=f'Mean={np.mean(edge_densities):.3f}')
    ax2.set_xlabel('Edge Density', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Per-Sample Edge Density', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edge_sparsity_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: edge_sparsity_analysis.png")
    
    return stats


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


def test_lens2_model(args):
    """
    Main testing function for LENS2 model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🧪 LENS2 MODEL TESTING")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Model: {args.model_path}")
    print(f"  Test Data: {args.test_data}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Load test data
    print("📂 Loading test dataset...")
    with open(args.test_data, 'r') as f:
        test_ids = f.readlines()
    
    test_dataset = GraphDataset(root=args.data_root, ids=test_ids)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate)
    
    print(f"   ✓ Loaded {len(test_dataset)} test samples")
    
    # Load model
    print("\n🔧 Loading model...")
    try:
        # Try with weights_only=False for backward compatibility
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract model configuration from checkpoint
    config = checkpoint.get('config', {})
    n_class = checkpoint.get('num_classes', args.n_class)
    
    # Determine if constrained or penalty mode
    use_constrained = config.get('use_constrained', False)
    l0_method = config.get('l0_method', 'hard-concrete')
    
    print(f"   Model Configuration:")
    print(f"   • Classes: {n_class}")
    print(f"   • Mode: {'CONSTRAINED' if use_constrained else 'PENALTY'}")
    print(f"   • L0 Method: {l0_method}")
    
    # Create model with same architecture
    model = ImprovedEdgeGNN(
        feature_dim=args.n_features,
        hidden_dim=args.hidden_dim,
        num_classes=n_class,
        # Architecture
        num_gnn_layers=config.get('num_gnn_layers', 3),
        num_attention_heads=config.get('num_attention_heads', 4),
        use_attention_pooling=config.get('use_attention_pooling', True),
        # Regularization
        lambda_reg=config.get('lambda_reg', args.lambda_reg),
        reg_mode=config.get('reg_mode', 'l0'),
        l0_method=l0_method,
        # Constrained
        use_constrained=use_constrained,
        constraint_target=config.get('constraint_target', 0.30),
        # Other
        edge_dim=args.edge_dim,
        dropout=args.dropout,
        warmup_epochs=args.warmup_epochs,
        graph_size_adaptation=args.graph_size_adaptation,
        min_edges_per_node=args.min_edges_per_node,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✓ Model loaded successfully")
    
    # Collect predictions
    evaluator = Evaluator(n_class=n_class)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("\n🔍 Collecting predictions...")
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Processing"):
            pred, labels, _, _ = evaluator.eval_test(
                sample, model, n_features=args.n_features
            )
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(pred.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    print(f"   ✓ Collected {len(all_labels)} predictions")
    
    # Calculate basic metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(all_labels, all_probabilities)
    
    print(f"\n   Overall Performance:")
    print(f"   • Accuracy: {metrics['accuracy']:.4f}")
    print(f"   • Macro F1: {metrics['f1_macro']:.4f}")
    print(f"   • Weighted F1: {metrics['f1_weighted']:.4f}")
    
    # Bootstrap analysis
    if args.n_bootstrap > 0:
        print(f"\n🔄 Running bootstrap analysis ({args.n_bootstrap} iterations)...")
        bootstrap_results = bootstrap_roc_pr_analysis(
            all_labels, 
            all_probabilities, 
            n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level
        )
    else:
        bootstrap_results = None
        print("\n⏭️  Skipping bootstrap analysis (--n-bootstrap 0)")
    
    # Analyze edge sparsity
    if args.analyze_sparsity:
        sparsity_stats = analyze_edge_sparsity(
            model, test_loader, device, args.n_features, args.output_dir
        )
    else:
        sparsity_stats = None
        print("\n⏭️  Skipping sparsity analysis (use --analyze-sparsity)")
    
    # Generate plots
    print("\n📈 Generating visualizations...")
    
    # Class names
    if args.class_names:
        class_names = args.class_names.split(',')
    else:
        class_names = [f'Class {i}' for i in range(n_class)]
    
    if bootstrap_results:
        # ROC/PR curves
        plot_combined_curves(
            bootstrap_results, 
            n_class, 
            os.path.join(args.output_dir, 'roc_pr_curves.png'),
            class_names=class_names
        )
    
    # Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Save comprehensive results
    print("\n💾 Saving results...")
    
    results_file = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LENS2 MODEL TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("Model Information:\n")
        f.write("-"*40 + "\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Test Data: {args.test_data}\n")
        f.write(f"Number of Samples: {len(all_labels)}\n")
        f.write(f"Number of Classes: {n_class}\n")
        f.write(f"Optimization Mode: {'CONSTRAINED' if use_constrained else 'PENALTY'}\n")
        f.write(f"L0 Method: {l0_method}\n")
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
            
            for i in range(n_class):
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
            
            f.write("\nMacro Average:\n")
            if 'macro' in bootstrap_results:
                res = bootstrap_results['macro']
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
            f.write(f"Median Edge Weight: {sparsity_stats['median_edge_weight']:.4f}\n")
            f.write(f"Edges > 0.1: {sparsity_stats['sparsity_01']:.2f}%\n")
            f.write(f"Edges > 0.5: {sparsity_stats['sparsity_05']:.2f}%\n")
            f.write(f"Edges > 0.9: {sparsity_stats['sparsity_09']:.2f}%\n")
            f.write(f"Avg Nodes per Graph: {sparsity_stats['avg_num_nodes']:.1f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write("-"*40 + "\n")
        f.write(str(metrics['confusion_matrix']) + "\n")
    
    # Save JSON summary
    json_results = {
        'accuracy': float(metrics['accuracy']),
        'f1_macro': float(metrics['f1_macro']),
        'f1_weighted': float(metrics['f1_weighted']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
    }
    
    if bootstrap_results and 'macro' in bootstrap_results:
        json_results['roc_auc_macro'] = float(bootstrap_results['macro']['roc_auc_mean'])
        json_results['pr_auc_macro'] = float(bootstrap_results['macro']['pr_auc_mean'])
    
    if sparsity_stats:
        json_results['edge_density'] = float(sparsity_stats['mean_density'])
        json_results['sparsity_01'] = float(sparsity_stats['sparsity_01'])
    
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
    
    return metrics, bootstrap_results, sparsity_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test LENS2 Model with Bootstrap Analysis')
    
    # Required
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data file')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory for dataset')
    
    # Model architecture
    parser.add_argument('--n-features', type=int, default=512, help='Number of node features')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--n-class', type=int, default=3, help='Number of classes (fallback)')
    parser.add_argument('--edge-dim', type=int, default=128, help='Edge dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Regularization (fallback values)
    parser.add_argument('--lambda-reg', type=float, default=0.01, help='Regularization strength')
    parser.add_argument('--warmup-epochs', type=int, default=15, help='Warmup epochs')
    parser.add_argument('--graph-size-adaptation', action='store_true', help='Graph size adaptation')
    parser.add_argument('--min-edges-per-node', type=float, default=2.0, help='Min edges per node')
    
    # Testing parameters
    parser.add_argument('--n-bootstrap', type=int, default=10000, 
                       help='Bootstrap iterations (0 to skip)')
    parser.add_argument('--confidence-level', type=float, default=0.95, 
                       help='Confidence level')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--analyze-sparsity', action='store_true', 
                       help='Analyze edge sparsity')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='test_results', help='Output directory')
    parser.add_argument('--class-names', type=str, default=None, 
                       help='Comma-separated class names (e.g., "Benign,Malignant,Normal")')
    
    args = parser.parse_args()
    
    # Run testing
    metrics, bootstrap_results, sparsity_stats = test_lens2_model(args)
