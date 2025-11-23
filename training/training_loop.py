"""
Training loop with WandB logging supporting BOTH modes:
- PENALTY MODE: Scheduled lambda with adaptive scaling and density loss
- CONSTRAINED MODE: Lagrangian optimization with dual variable updates

Enhanced with phase-specific model saving (warmup, ramp, plateau) and sparsity matrices

FIXED: Dual variable updates moved from per-batch to per-epoch (prevents collapse)
"""

import os
import torch
import gc
import wandb
import numpy as np
from helper import Trainer, Evaluator, preparefeatureLabel
from model.EGL_L0_Reg import compute_density


def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device,
                      num_epochs, n_features, output_dir, warmup_epochs=5,
                      ramp_epochs=20, wandb_config=None, l0_method='hard-concrete'):
    """
    Train and evaluate with phase-specific model saving
    """
    # Initialize WandB if config provided
    use_wandb = wandb_config is not None
    if use_wandb:
        wandb.init(**wandb_config)
        wandb.watch(model, log='all', log_freq=100)
    
    # Initialize trainer and evaluator
    trainer = Trainer(n_class=model.num_classes)
    evaluator = Evaluator(n_class=model.num_classes)
    
    # Create output directories
    adj_output_dir = os.path.join(output_dir, 'pruned_adjacencies')
    matrices_dir = os.path.join(output_dir, 'sparsity_matrices')
    models_dir = os.path.join(output_dir, 'phase_models')
    
    for dir_path in [adj_output_dir, matrices_dir, models_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Track best models for each phase
    best_warmup = {'acc': 0.0, 'epoch': 0, 'density': 0.0}
    best_ramp = {'acc': 0.0, 'epoch': 0, 'density': 0.0}
    best_plateau = {'acc': 0.0, 'epoch': 0, 'density': 0.0}
    best_overall = {'acc': 0.0, 'epoch': 0, 'density': 0.0, 'phase': ''}
    
    # Track metrics
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    density_history = []
    constraint_violation_history = []
    dual_lambda_history = []
    
    # Determine mode
    use_constrained = hasattr(model.regularizer, 'use_constrained') and model.regularizer.use_constrained
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING WITH PHASE-SPECIFIC MODEL SAVING")
    print(f"   Mode: {'CONSTRAINED' if use_constrained else 'PENALTY'}")
    print(f"   Will save best models from: warmup, ramp, plateau phases")
    if use_constrained:
        print(f"   🔧 Dual variable updates: PER-EPOCH (fixed)")
    print(f"{'='*70}\n")
    
    # Main training loop
    for epoch in range(num_epochs):
        # Determine current phase
        if epoch < warmup_epochs:
            current_phase = "warmup"
            phase_color = "🟡"
        elif epoch < warmup_epochs + ramp_epochs:
            current_phase = "ramp"
            phase_color = "🟠"
        else:
            current_phase = "plateau"
            phase_color = "🟢"
        
        print(f"\n{'='*60}")
        print(f"{phase_color} Epoch {epoch+1}/{num_epochs} - Phase: {current_phase.upper()}")
        print(f"{'='*60}")
        
        # Training phase
        train_metrics = train_epoch(
            epoch, model, train_loader, optimizer, scheduler,
            trainer, n_features, num_epochs, warmup_epochs, ramp_epochs,
            matrices_dir, l0_method
        )
        
        train_accs.append(train_metrics['accuracy'])
        train_losses.append(train_metrics['loss'])
        density_history.append(train_metrics.get('current_density', 0))
        
        if use_constrained:
            constraint_violation_history.append(train_metrics.get('avg_constraint_violation', 0))
            dual_lambda_history.append(train_metrics.get('dual_lambda', 0))
        
        # Log to WandB
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'phase': current_phase,
                'train/accuracy': train_metrics['accuracy'],
                'train/loss': train_metrics['loss'],
                'train/cls_loss': train_metrics.get('cls_loss', 0),
                'train/reg_loss': train_metrics.get('reg_loss', 0),
                'density/current': train_metrics.get('current_density', 0),
                'hyperparams/temperature': train_metrics.get('temperature', 0),
                'hyperparams/learning_rate': optimizer.param_groups[0]['lr'],
            }
            
            if use_constrained:
                log_dict.update({
                    'constrained/dual_lambda': train_metrics.get('dual_lambda', 0),
                    'constrained/constraint_violation': train_metrics.get('avg_constraint_violation', 0),
                })
            else:
                log_dict.update({
                    'penalty/lambda_eff': train_metrics.get('lambda_eff', 0),
                    'penalty/alpha': train_metrics.get('avg_alpha', 1.0),
                })
            
            wandb.log(log_dict)
        
        torch.cuda.empty_cache()
        
        # Validation phase
        val_metrics, edge_weights_sample = validate_epoch_with_matrices(
            model, val_loader, evaluator, n_features, epoch
        )
        
        val_accs.append(val_metrics['accuracy'])
        val_losses.append(val_metrics['loss'])
        
        print(f"\n   🔹 Validation Accuracy: {val_metrics['accuracy']:.4f}, "
              f"Loss: {val_metrics['loss']:.4f}")
        print(f"   🔹 Current Density: {train_metrics.get('current_density', 0)*100:.1f}%")
        
        # ====================================================================
        # PHASE-SPECIFIC MODEL SAVING
        # ====================================================================
        val_acc = val_metrics['accuracy']
        current_density = train_metrics.get('current_density', 0)
        
        # Check if best for current phase
        save_model = False
        update_phase = None
        
        if current_phase == "warmup":
            if val_acc > best_warmup['acc']:
                best_warmup.update({'acc': val_acc, 'epoch': epoch+1, 'density': current_density})
                save_model = True
                update_phase = "warmup"
                
        elif current_phase == "ramp":
            if val_acc > best_ramp['acc']:
                best_ramp.update({'acc': val_acc, 'epoch': epoch+1, 'density': current_density})
                save_model = True
                update_phase = "ramp"
                
        elif current_phase == "plateau":
            if val_acc > best_plateau['acc']:
                best_plateau.update({'acc': val_acc, 'epoch': epoch+1, 'density': current_density})
                save_model = True
                update_phase = "plateau"
        
        # Check if overall best
        if val_acc > best_overall['acc']:
            best_overall.update({
                'acc': val_acc, 
                'epoch': epoch+1, 
                'density': current_density,
                'phase': current_phase
            })
            # Save overall best
            save_model_with_matrices(
                model, optimizer, epoch+1, val_acc, current_density,
                models_dir, matrices_dir, phase="overall_best",
                edge_weights=edge_weights_sample, train_metrics=train_metrics,
                use_constrained=use_constrained
            )
            print(f"   ✨ NEW OVERALL BEST! Acc: {val_acc:.4f}, Density: {current_density*100:.1f}%")
        
        # Save phase-specific best
        if save_model and update_phase:
            save_model_with_matrices(
                model, optimizer, epoch+1, val_acc, current_density,
                models_dir, matrices_dir, phase=f"best_{update_phase}",
                edge_weights=edge_weights_sample, train_metrics=train_metrics,
                use_constrained=use_constrained
            )
            print(f"   ✅ NEW {update_phase.upper()} BEST! Acc: {val_acc:.4f}, Density: {current_density*100:.1f}%")
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                'val/accuracy': val_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/edge_sparsity': val_metrics.get('edge_sparsity', 0),
                f'best/{current_phase}_accuracy': best_warmup['acc'] if current_phase == 'warmup' else (
                    best_ramp['acc'] if current_phase == 'ramp' else best_plateau['acc']
                ),
                'best/overall_accuracy': best_overall['acc'],
            })
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Best Overall: Acc={best_overall['acc']:.4f} at epoch {best_overall['epoch']} ({best_overall['phase']} phase)")
    print(f"  Density: {best_overall['density']*100:.1f}%")
    print(f"\nBest by Phase:")
    print(f"  Warmup:  Acc={best_warmup['acc']:.4f} at epoch {best_warmup['epoch']}, Density={best_warmup['density']*100:.1f}%")
    print(f"  Ramp:    Acc={best_ramp['acc']:.4f} at epoch {best_ramp['epoch']}, Density={best_ramp['density']*100:.1f}%")
    print(f"  Plateau: Acc={best_plateau['acc']:.4f} at epoch {best_plateau['epoch']}, Density={best_plateau['density']*100:.1f}%")
    print(f"{'='*70}\n")
    
    # Finish WandB
    if use_wandb:
        wandb.finish()
    
    return {
        "train_accs": train_accs,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_losses": val_losses,
        "density_history": density_history,
        "constraint_violation_history": constraint_violation_history if use_constrained else [],
        "dual_lambda_history": dual_lambda_history if use_constrained else [],
        "best_val_acc": best_overall['acc'],
        "best_epoch": best_overall['epoch'],
        "best_warmup": best_warmup,
        "best_ramp": best_ramp,
        "best_plateau": best_plateau,
    }


def save_model_with_matrices(model, optimizer, epoch, val_acc, current_density,
                            models_dir, matrices_dir, phase="best",
                            edge_weights=None, train_metrics=None,
                            use_constrained=False):
    """
    Save model checkpoint with associated metrics and matrices
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'current_density': current_density,
        'edge_density': current_density,
        'temperature': model.temperature if hasattr(model, 'temperature') else 1.0,
        'l0_method': model.l0_method if hasattr(model, 'l0_method') else 'hard-concrete',
        'use_constrained': use_constrained,
    }
    
    # Add train metrics if available
    if train_metrics:
        checkpoint['train_metrics'] = train_metrics
    
    # Save model
    model_path = os.path.join(models_dir, f'{phase}_model.pth')
    torch.save(checkpoint, model_path)
    
    # Save edge weight matrix if available
    if edge_weights is not None:
        matrix_path = os.path.join(matrices_dir, f'{phase}_edge_weights.pt')
        torch.save(edge_weights, matrix_path)


def validate_epoch_with_matrices(model, val_loader, evaluator, n_features, epoch):
    """
    Validation with edge weight extraction
    
    Returns:
        metrics: Validation metrics dict
        edge_weights_sample: Sample edge weights for saving
    """
    model.eval()
    val_loss = 0.0
    evaluator.reset_metrics()
    
    if hasattr(model, 'set_print_stats'):
        model.set_print_stats(False)
    
    all_edge_weights = []
    edge_weights_sample = None
    
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            try:
                pred, labels, loss, weighted_adj = evaluator.eval_test(
                    sample, model, n_features=n_features
                )
                val_loss += loss.item()
                
                # Get edge weights for first batch as sample
                if idx == 0 and weighted_adj is not None:
                    node_feat, labels, adjs, masks = preparefeatureLabel(
                        sample['image'], sample['label'], sample['adj_s'], n_features=n_features
                    )
                    # Normalize by original adjacency
                    edge_mask = (adjs > 0).float()
                    edge_weights = weighted_adj / (adjs + 1e-8)
                    edge_weights = edge_weights * edge_mask
                    edge_weights_sample = edge_weights[0]  # First in batch
                
                # Collect for statistics
                if weighted_adj is not None:
                    all_edge_weights.append(weighted_adj.detach().cpu())
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
    
    # Compute metrics
    val_acc = evaluator.get_scores()
    avg_val_loss = val_loss / max(1, len(val_loader))
    
    # Calculate edge sparsity
    edge_sparsity = 0.0
    if all_edge_weights:
        all_weights = torch.cat([w.flatten() for w in all_edge_weights])
        all_weights = all_weights[all_weights > 0]
        if len(all_weights) > 0:
            edge_sparsity = (all_weights > 0.1).float().mean().item() * 100.0
    
    metrics = {
        'accuracy': val_acc,
        'loss': avg_val_loss,
        'edge_sparsity': edge_sparsity,
    }
    
    return metrics, edge_weights_sample


def train_epoch(epoch, model, train_loader, optimizer, scheduler, trainer,
               n_features, num_epochs, warmup_epochs, ramp_epochs,
               matrices_dir, l0_method='hard-concrete'):
    """
    Run one epoch of training
    
    ✅ FIXED: Dual variable updates per-batch (as in paper Eq. 5)
    - Each batch computes its own constraint violation
    - Dual variable is updated IMMEDIATELY after each batch
    - No averaging of violations across batches
    
    Returns:
        dict: Training metrics including density and constraint info
    """
    model.set_epoch(epoch)
    
    # Determine optimization mode
    use_constrained = hasattr(model.regularizer, 'use_constrained') and model.regularizer.use_constrained
    
    # Update schedules (temperature, lambda for penalty mode)
    if hasattr(model.regularizer, 'update_all_schedules'):
        schedules = model.regularizer.update_all_schedules(
            current_epoch=epoch,
            initial_temp=model.initial_temp if hasattr(model, 'initial_temp') else 5.0
        )
        model.temperature = schedules['temperature']
    
    model.train()
    train_loss = 0.0
    cls_loss_sum = 0.0
    reg_loss_sum = 0.0
    trainer.reset_metrics()
    
    # Tracking lists (for logging/monitoring only)
    batch_densities = []
    constraint_violations = []  # For logging only - NOT used for updates
    dual_lambda_values = []     # Track dual lambda evolution
    alpha_values = []           # For penalty mode adaptive scaling
    lambda_eff_values = []      # For penalty mode effective lambda
    
    # Batch loop
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Control verbose printing
        if hasattr(model, 'set_print_stats'):
            model.set_print_stats(batch_idx % 50 == 0)
        
        # ====================================================================
        # FORWARD PASS
        # ====================================================================
        try:
            pred, labels, loss, weighted_adj = trainer.train(
                sample, model, n_features=n_features
            )
            
            # ================================================================
            # COLLECT BATCH STATISTICS (from model's stats tracker)
            # ================================================================
            if hasattr(model, 'stats_tracker'):
                # Classification and regularization losses
                if hasattr(model.stats_tracker, 'cls_loss_history') and len(model.stats_tracker.cls_loss_history) > 0:
                    cls_loss_sum += model.stats_tracker.cls_loss_history[-1]
                    reg_loss_sum += model.stats_tracker.reg_loss_history[-1]
                
                # Current density
                if hasattr(model.stats_tracker, 'current_density_history') and len(model.stats_tracker.current_density_history) > 0:
                    batch_densities.append(model.stats_tracker.current_density_history[-1])
                
                # Penalty mode specific metrics
                if not use_constrained:
                    if hasattr(model.stats_tracker, 'lambda_eff_history'):
                        if len(model.stats_tracker.lambda_eff_history) > 0:
                            lambda_eff_values.append(model.stats_tracker.lambda_eff_history[-1])
                            # Compute adaptive alpha
                            if model.regularizer.current_lambda > 0:
                                alpha = model.stats_tracker.lambda_eff_history[-1] / model.regularizer.current_lambda
                                alpha_values.append(alpha)
            
            # Check for NaN/Inf (skip batch if detected)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   ⚠️  Skipping batch {batch_idx}: NaN/Inf loss detected")
                continue
            
            # ================================================================
            # BACKWARD PASS
            # ================================================================
            loss.backward()
            
            # ================================================================
            # ✅ CONSTRAINED MODE: UPDATE DUAL VARIABLE PER-BATCH
            # ================================================================
            if use_constrained:
                # Check if model has computed constraint violation for this batch
                if hasattr(model, 'last_constraint_violation'):
                    if model.last_constraint_violation is not None:
                        # Get CURRENT batch's violation (not averaged!)
                        current_violation = model.last_constraint_violation
                        
                        # Paper Equation 5: λ^{t+1} = λ^t + η_dual * (g_const - ε)
                        # Paper Equation 6: With dual restart heuristic
                        model.regularizer.update_dual_variable(
                            current_violation  # ✅ Use current batch violation directly!
                        )
                        
                        # Store for logging/monitoring (not used for updates)
                        constraint_violations.append(current_violation)
                        dual_lambda_values.append(model.regularizer.dual_lambda)
                        
                        # Verbose logging for first few batches
                        if batch_idx < 5 and epoch < 3:
                            print(f"   [Batch {batch_idx}] violation={current_violation:.6f}, "
                                  f"λ_dual={model.regularizer.dual_lambda:.6f}")
            
            # ================================================================
            # GRADIENT CLIPPING
            # ================================================================
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # ================================================================
            # OPTIMIZER STEP
            # ================================================================
            optimizer.step()
            scheduler(optimizer, batch_idx, epoch, 0)
            
            train_loss += loss.item()
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"   ⚠️  CUDA OOM at batch {batch_idx}, clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
    
    # ========================================================================
    # COMPUTE EPOCH METRICS
    # ========================================================================
    train_acc = trainer.get_scores()
    avg_train_loss = train_loss / max(1, len(train_loader))
    avg_cls_loss = cls_loss_sum / max(1, len(train_loader))
    avg_reg_loss = reg_loss_sum / max(1, len(train_loader))
    avg_density = np.mean(batch_densities) if batch_densities else 0.0
    
    # Base metrics (common to both modes)
    metrics = {
        'accuracy': train_acc,
        'loss': avg_train_loss,
        'cls_loss': avg_cls_loss,
        'reg_loss': avg_reg_loss,
        'current_density': avg_density,
        'temperature': model.temperature if hasattr(model, 'temperature') else 0.0,
        'current_lambda': model.regularizer.current_lambda if hasattr(model.regularizer, 'current_lambda') else 0.0,
    }
    
    # ========================================================================
    # MODE-SPECIFIC METRICS
    # ========================================================================
    if use_constrained:
        # Constrained mode metrics
        # Average violation FOR LOGGING ONLY (not used for updates!)
        avg_violation = np.mean(constraint_violations) if constraint_violations else 0.0
        
        # Get final dual lambda value (after all batch updates)
        final_dual_lambda = dual_lambda_values[-1] if dual_lambda_values else model.regularizer.dual_lambda
        
        # Check how many batches satisfied constraint
        num_satisfied = sum(1 for v in constraint_violations if v <= 0)
        satisfaction_rate = num_satisfied / max(1, len(constraint_violations))
        
        metrics.update({
            'dual_lambda': final_dual_lambda,
            'avg_constraint_violation': avg_violation,  # For logging only
            'lambda_eff': final_dual_lambda,  # Effective lambda is dual lambda
            'constraint_satisfaction_rate': satisfaction_rate,
            'num_batches_satisfied': num_satisfied,
            'total_batches': len(constraint_violations),
        })
        
        # Print epoch summary
        print(f"\n   📊 Epoch {epoch+1} Summary (CONSTRAINED MODE):")
        print(f"      Loss: {avg_train_loss:.4f} (cls: {avg_cls_loss:.4f}, reg: {avg_reg_loss:.4f})")
        print(f"      Accuracy: {train_acc:.4f}")
        print(f"      Density: {avg_density*100:.2f}% (target: {model.regularizer.constraint_target*100:.1f}%)")
        print(f"      λ_dual: {final_dual_lambda:.6f}")
        print(f"      Avg Violation: {avg_violation:.6f}")
        print(f"      Satisfaction Rate: {satisfaction_rate*100:.1f}%")
        
    else:
        # Penalty mode metrics
        avg_alpha = np.mean(alpha_values) if alpha_values else 1.0
        avg_lambda_eff = np.mean(lambda_eff_values) if lambda_eff_values else 0.0
        
        metrics.update({
            'avg_alpha': avg_alpha,
            'lambda_eff': avg_lambda_eff,
            'lambda_density': model.regularizer.current_lambda_density if hasattr(model.regularizer, 'current_lambda_density') else 0.0,
            'target_density': model.regularizer.target_density if hasattr(model.regularizer, 'target_density') else 0.0,
            'density_deviation': abs(avg_density - model.regularizer.target_density) if hasattr(model.regularizer, 'target_density') else 0.0,
        })
        
        # Print epoch summary
        print(f"\n   📊 Epoch {epoch+1} Summary (PENALTY MODE):")
        print(f"      Loss: {avg_train_loss:.4f} (cls: {avg_cls_loss:.4f}, reg: {avg_reg_loss:.4f})")
        print(f"      Accuracy: {train_acc:.4f}")
        print(f"      Density: {avg_density*100:.2f}% (target: {model.regularizer.target_density*100:.1f}%)")
        print(f"      λ_eff: {avg_lambda_eff:.6f} (α: {avg_alpha:.3f})")
    
    return metrics
