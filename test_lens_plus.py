#!/usr/bin/env python
"""
Test script for LENS model with bootstrap confidence intervals
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report,
                           accuracy_score, f1_score, precision_score, recall_score)
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

from utils.dataset import GraphDataset
from model.LENS2 import ImprovedEdgeGNN
from helper import Evaluator, collate, preparefeatureLabel

def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device,weights_only=False)
    print(f"Model from epoch {checkpoint['epoch']} with val_acc={checkpoint['val_accuracy']:.4f}")
    
    # Initialize model
    model = ImprovedEdgeGNN(
        feature_dim=512,
        hidden_dim=256,
        num_classes=args.n_class,
        lambda_reg=0.0001,
        reg_mode='l0',
        l0_method=checkpoint.get('l0_method', 'hard-concrete'),
        edge_dim=args.edge_dim,
        dropout=0,
        l0_beta=1,
        num_gnn_layers=args.num_gnn_layers,
        num_attention_heads=args.num_attention_heads,
        use_attention_pooling=args.use_attention_pooling,
        use_constrained=checkpoint.get('use_constrained', False),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    # After line 45 in test_lens_plus.py

    # ✅ SET THE SAME EPOCH AND TEMPERATURE AS VALIDATION
    model.current_epoch = checkpoint['epoch']
    model.regularizer.current_epoch = checkpoint['epoch']

    # Restore temperature (should be ~1.0 at epoch 148)
    if 'temperature' in checkpoint:
        model.temperature = checkpoint['temperature']
    else:
       # Calculate what temperature was at that epoch
       schedules = model.regularizer.update_all_schedules(
        current_epoch=checkpoint['epoch'],
        initial_temp=5.0  # Your initial_temp
       )
       model.temperature = schedules['temperature']

    print(f"✅ Restored: epoch={model.current_epoch}, temperature={model.temperature:.4f}")

    model.eval()
    print(f"Loading test data from {args.test_list}")
    with open(args.test_list, 'r') as f:
        test_ids = f.readlines()
    # Load test data
    dataset = GraphDataset(root=args.data_root, ids=test_ids)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    
    print(f"\nTesting on {len(test_loader)} samples...")
    # After model.eval() and before testing
    print("\n🔍 Checking LogAlpha distribution:")
    with torch.no_grad():
      sample = next(iter(test_loader))
      node_feat, labels, adjs, masks = preparefeatureLabel(
        sample['image'], sample['label'], sample['adj_s'], n_features=512
      )
    
      edge_weights, logAlpha = model.edge_scorer.compute_edge_weights(
        node_feat, adjs, training=False, temperature=1.0,
        l0_params=model.l0_params, use_l0=True
      )
    
      edge_mask = (adjs > 0)
      valid_logAlpha = logAlpha[edge_mask]
    
      print(f"LogAlpha stats (valid edges only):")
      print(f"  Min: {valid_logAlpha.min():.4f}")
      print(f"  Max: {valid_logAlpha.max():.4f}")
      print(f"  Mean: {valid_logAlpha.mean():.4f}")
      print(f"  Std: {valid_logAlpha.std():.4f}")
    
      # Check keep probabilities
      keep_probs = torch.sigmoid(valid_logAlpha - model.l0_params.const1)
      print(f"\nKeep probabilities:")
      print(f"  Min: {keep_probs.min():.4f}")
      print(f"  Max: {keep_probs.max():.4f}")
      print(f"  Mean: {keep_probs.mean():.4f}")
      print(f"  Fraction > 0.5: {(keep_probs > 0.5).float().mean():.4f}")
    # Initialize evaluator
    evaluator = Evaluator(n_class=args.n_class)
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    edge_densities = []
    test_loss = 0.0
    
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_loader, desc="Testing")):
            # Forward pass
            pred, labels, loss, weighted_adj = evaluator.eval_test(
                sample, model, n_features=512
            )
            
            # Store results
            probs = pred  # Already softmax from evaluator
            pred_class = torch.argmax(probs, dim=1)
            
            all_preds.extend(pred_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            cls_loss = torch.nn.functional.cross_entropy(pred, labels)
            test_loss += cls_loss.item()            
            # Calculate edge density
            node_feat, labels, adjs, masks = preparefeatureLabel(
                sample['image'], sample['label'], sample['adj_s'], n_features=512
            )
            edge_mask = (adjs > 0).float()
            if edge_mask.sum() > 0:
                density = (weighted_adj * edge_mask).sum() / edge_mask.sum()
                edge_densities.append(density.item())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    avg_loss = test_loss / len(test_loader)
    avg_density = np.mean(edge_densities) if edge_densities else 0
    
    # ROC AUC for binary classification
    if args.n_class == 2:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    class_names = [f'Class_{i}' for i in range(args.n_class)]
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    if args.n_class == 2:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Avg Edge Density: {avg_density*100:.2f}%")
    print(f"\n{report}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n")
        f.write(f"Val Accuracy (saved): {checkpoint['val_accuracy']:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score (macro): {f1_macro:.4f}\n")
        f.write(f"Precision (macro): {precision_macro:.4f}\n")
        f.write(f"Recall (macro): {recall_macro:.4f}\n")
        if args.n_class == 2:
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Avg Edge Density: {avg_density*100:.2f}%\n")
        f.write(f"Edge Density (saved): {checkpoint.get('edge_density', checkpoint.get('current_density', 0))*100:.2f}%\n")
        f.write(f"\n{report}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save predictions
    np.savez(os.path.join(args.output_dir, 'predictions.npz'),
             predictions=all_preds,
             labels=all_labels,
             probabilities=all_probs,
             edge_densities=edge_densities)
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Bootstrap analysis if requested
    if args.bootstrap:
        print(f"\nPerforming bootstrap analysis with {args.n_bootstrap} iterations...")
        bootstrap_ci = bootstrap_confidence_intervals(all_labels, all_probs, args.n_bootstrap)
        
        with open(os.path.join(args.output_dir, 'bootstrap_results.txt'), 'w') as f:
            f.write("Bootstrap 95% Confidence Intervals:\n")
            f.write(f"Accuracy: {bootstrap_ci['acc_mean']:.4f} [{bootstrap_ci['acc_low']:.4f}, {bootstrap_ci['acc_high']:.4f}]\n")
            f.write(f"F1 Score: {bootstrap_ci['f1_mean']:.4f} [{bootstrap_ci['f1_low']:.4f}, {bootstrap_ci['f1_high']:.4f}]\n")
            if args.n_class == 2:
                f.write(f"ROC AUC: {bootstrap_ci['auc_mean']:.4f} [{bootstrap_ci['auc_low']:.4f}, {bootstrap_ci['auc_high']:.4f}]\n")

def bootstrap_confidence_intervals(y_true, y_probs, n_bootstrap=1000):
    """Simple bootstrap CI calculation"""
    n_samples = len(y_true)
    
    accs = []
    f1s = []
    aucs = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_probs_boot = y_probs[indices]
        y_pred_boot = np.argmax(y_probs_boot, axis=1)
        
        accs.append(accuracy_score(y_true_boot, y_pred_boot))
        f1s.append(f1_score(y_true_boot, y_pred_boot, average='macro'))
        
        if y_probs.shape[1] == 2:
            aucs.append(roc_auc_score(y_true_boot, y_probs_boot[:, 1]))
    
    return {
        'acc_mean': np.mean(accs),
        'acc_low': np.percentile(accs, 2.5),
        'acc_high': np.percentile(accs, 97.5),
        'f1_mean': np.mean(f1s),
        'f1_low': np.percentile(f1s, 2.5),
        'f1_high': np.percentile(f1s, 97.5),
        'auc_mean': np.mean(aucs) if aucs else 0,
        'auc_low': np.percentile(aucs, 2.5) if aucs else 0,
        'auc_high': np.percentile(aucs, 97.5) if aucs else 0,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--test-list', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./test_results')
    parser.add_argument('--n-class', type=int, default=2)
    parser.add_argument('--num-gnn-layers', type=int, default=3)
    parser.add_argument('--num-attention-heads', type=int, default=8)
    parser.add_argument('--use-attention-pooling', action='store_true')
    parser.add_argument('--edge-dim', type=int, default=128)
    parser.add_argument('--bootstrap', action='store_true', help='Perform bootstrap CI')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    
    args = parser.parse_args()
    test_model(args)
