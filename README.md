<!-- Project Title Banner -->
<h1 align="center">LENS: Learnable Edge Network Sparsification for Interpretable Histopathology</h1>
<!-- Overview Image -->
<div align="center">
<img width="1436" alt="Image" src="https://github.com/user-attachments/assets/0aa6df98-e016-404d-9b3a-f1612f386aab" /> 
  <p><em>LENS: A graph neural network approach for interpretable histopathology analysis through learnable edge sparsification</em></p>
</div>



##  Pipeline Overview

LENS offers a complete pipeline for histopathology analysis, from whole slide images to interpretable graph-based classification:
<table align="center">
<tr>
<td align="center" width="20%">

<a href="#step-1-wsi-tiling">
<img width="100"  src="https://github.com/user-attachments/assets/ae412056-6dda-4d9c-9d34-a30e0a2fcafd"/>
<br/><strong>Step 1</strong><br/>
 <strong>WSI Tiling</strong><br/>
<em>Partition slides into manageable patches</em>
</a>
</td>
<td align="center" width="20%">

<a href="#step-2-graph-construction">
<img width="80" src="https://github.com/user-attachments/assets/e4b5f7cf-0efd-4772-8fcb-807b2e923592"/>
<br/><strong>Step 2</strong><br/>
 <strong>Graph Construction</strong><br/>
<em>Build spatial relationships between patches</em>
</a>
</td>
<td align="center" width="20%">
<a href="#step-3-model-training">
<img width="50" src="https://img.icons8.com/fluency/96/000000/artificial-intelligence.png"/>
<br/><strong>Step 3</strong><br/>
 <strong>Model Training</strong><br/>
<em>Learn sparse, interpretable representations</em>
</a>
</td>

<td align="center" width="20%">
<a href="#step-4-testing">
<img width="50" src="https://github.com/user-attachments/assets/da28dbeb-7fff-4c49-88a0-de8fab04b10c"/>
<br/><strong>Step 4</strong><br/>
 <strong>Testing</strong><br/>
<em>Evaluate with bootstrap statistics</em>
</a>

</td>

<td align="center" width="20%">
<a href="#step-5-visualization">
<img width="80" src="https://github.com/user-attachments/assets/8392eb50-bee5-438d-9ab7-6d61e893f142"/>
<br/><strong>Step 5</strong><br/>
 <strong>Visualization</strong><br/>
<em>Generate interpretable heatmaps</em>
</a>
</td>
</tr>
</table>
## 📦 Installation

```bash
# Clone the repository
https://github.com/LENS-network-ai/LENS-framework-.git
cd LENS

# Install dependencies
pip install -r requirements.txt
```

### Dataset Information
Our experiments use the **CPTAC** (Clinical Proteomic Tumor Analysis Consortium) dataset, which is publicly available through the National Cancer Institute.
Visit: https://www.cancerimagingarchive.net/
You find the IDs used in this study in CPTAC_IDs.txt

### Step 1: WSI Tiling
```bash
# -s 512      # Tile size: 512x512 pixels  
#-e 0        # Overlap: 0px (no extra pixels added on edges)  
#-j 32       # Threads: use 32 parallel threads  
#-B 50       # Max background: skip tiles with >50% background  
#-o [path]   # Output path: where to save the tiles  
#-M -1       # Magnification: -1 = all levels, or set a specific one  

python tiling.py -s 512 -e 0 -j 32 -B 50 -M 20 -o <full_patch_to_output_folder> "full_path_to_input_slides/*/*.svs"
```

### Step 2: Graph Construction
```python
# Create a graph dataset from the tiled images

python graph_construction.py  --weights "path_to_pretrained_feature_extractor" --dataset "path_to_patches" --output "../graphs"
```
In our work we have mainly used the pretrained feature extractor from GTP work: [ResNetSimCLR](https://github.com/vkola-lab/tmi2022/tree/main/feature_extractor)

**Expected structure**:
```
    data/
    ├── slides/                    # Original .svs files (optional for training)
    ├── graphs/LUAD/simclr_files/  # Preprocessed graph data
    │   ├── C3N-03093-21/          # CPTAC sample ID
    │   │   ├── adj_s.pt           # Adjacency matrix [N×N]
    │   │   ├── features.pt        # Node features [N×512] 
    │   │   └── c_idx.txt          # Patch coordinates
    │   ├── C3N-01179-21/
    │   │   └── ...
    │   └── C3N-01334-21/
    │       └── ...
    ├── trainVal5.txt              # Training/validation samples 70% (CPTAC IDs)
    └── test_list.txt              # Test samples 30 %(CPTAC IDs)
```
### Step 3: Model Training
There are two ways to run LENS model:

1. **Standard Training**: Train the model with fixed hyperparameters
2. **Bayesian Optimization**: Automatically find optimal hyperparameters based on validation accuracy and sparsity

### Standard Training

To run standard training with cross-validation:

```bash
python main.py \
  --data-root /path/to/data \
  --train-list /path/to/train_list.txt \
  --batch-size 1 \
  --epochs 80 \
  --lambda-reg 0.00001 \
  --reg-mode l0 \
  --warmup-epochs 60 \
  --min-edges-per-node 2
```

#### Key Parameters

- `--data-root`: Directory containing the graph data
- `--train-list`: File with list of training examples
- `--lambda-reg`: Regularization strength (λ) controlling sparsity (lower = less pruning)
- `--reg-mode`: Regularization type (`l0` or `egl`)
- `--warmup-epochs`: Number of epochs with gradually increasing regularization
- `--min-edges-per-node`: Minimum edges to maintain per node

#### L0 Specific Parameters

When using L0 regularization, you can customize the following:

```bash
python main.py \
  --data-root /path/to/data \
  --train-list /path/to/train_list.txt \
  --reg-mode l0 \
  --lambda-reg 0.00001 \
  --l0-gamma -0.1 \
  --l0-zeta 1.1 \
  --l0-beta 0.66 \
  --initial-temp 5.0
```

- `--l0-gamma`: Lower bound of hard sigmoid (default: -0.1)
- `--l0-zeta`: Upper bound of hard sigmoid (default: 1.1)
- `--l0-beta`: Temperature parameter for L0 regularization (default: 0.66)
- `--initial-temp`: Initial temperature for edge gating (default: 5.0)

### Bayesian Optimization

To automatically find optimal parameters balancing accuracy and sparsity:

```bash
python main.py \
  --data-root /path/to/data \
  --train-list /path/to/train_list.txt \
  --batch-size 1 \
  --epochs 80 \
  --run-bayesian-opt \
  --n-trials 30 \
  --target-sparsity 0.7 \
  --sparsity-penalty 5.0
```

#### Optimization Parameters

- `--run-bayesian-opt`: Flag to activate Bayesian optimization
- `--n-trials`: Number of optimization trials to run (default: 50)
- `--target-sparsity`: Target sparsity rate to aim for (0.0-1.0)
- `--sparsity-penalty`: Weight for sparsity deviation penalty (higher = stricter adherence to target)

The optimization objective is: `O = Accuracy − sparsity_penalty · |SparsityRate − Target|`

#### Optimized Parameters

The Bayesian optimization will search for optimal values of:
- `lambda_reg`: Regularization strength
- `warmup_epochs`: Number of warmup epochs
- `l0_gamma`, `l0_zeta`, `l0_beta`: L0 regularization parameters
- `initial_temp`: Temperature parameter

### Output

Both training modes will create an output directory with:
- Model checkpoints
- Training metrics
- Graph visualizations
- Sparsification reports
- Edge weight distributions

For Bayesian optimization, additional outputs include:
- Parameter importance analysis
- Optimization history plots
- Detailed trial results


## Graph Sparsification Process

During training, the model goes through the following stages:

1. **Warmup Phase** (controlled by `warmup_epochs`):
   - Regularization strength gradually increases
   - All edges are initially kept to learn basic representations
   - Temperature parameter starts high for exploration

2. **Sparsification Phase**:
   - Full regularization applied
   - Edge weights shift toward binary (0 or 1)
   - Temperature annealing to sharpen decisions

3. **Refinement Phase**:
   - Fine-tuning of edge weights
   - Optimization focuses on important connections
   - Sparsity pattern stabilizes

## Cross-Validation

The model automatically performs cross-validation using the specified number of folds:

```bash
python main.py \
  --data-root /path/to/data \
  --train-list /path/to/train_list.txt \
  --n-folds 5 \
  --reg-mode l0 \
  --lambda-reg 0.00001
```

After training completes, detailed cross-validation results are displayed and saved, including:
- Average validation accuracy across folds
- Average edge sparsity
- Variance in performance
- Overfitting analysis
- Recommendations for hyperparameter adjustments

<img src="https://github.com/user-attachments/assets/eee7e786-9605-4c5a-b032-c5abd81998db" width="400"/>

### Step 4: Testing



The testing script evaluates trained models with bootstrap statistical analysis, providing confidence intervals for ROC/PR curves and saving weighted adjacency matrices for visualization.

### Usage

    python Test.py \
      --model-path /path/to/pretrained/model.pt \
      --test-data /path/to/test/data.txt \
      --data-root /path/to/graph/data \
      --lambda-reg 0.000182 \
      --reg-mode l0 \
      --l0-gamma -0.12 \
      --l0-zeta 1.09 \
      --l0-beta 0.72 \
      --n-bootstrap 10000 \
      --output-dir test_results

### Key Parameters

- `--model-path`: Path to trained model checkpoint
- `--test-data`: Text file with test sample IDs  
- `--lambda-reg`: Regularization strength (use optimized value)
- `--reg-mode`: Regularization type (l0 recommended)
- `--n-bootstrap`: Bootstrap iterations for confidence intervals (default: 10000)

### Output

The script generates three types of outputs:

#### Statistical Results
    test_results/comprehensive_results.txt    # Detailed metrics with bootstrap CI

#### Visualization Files  
    test_results/roc_pr_curves_combined.png   # ROC/PR curves for all classes
    test_results/roc_pr_class_0.png           # Individual class curves

#### Weighted Adjacencies
    test_results/weighted_adjacencies/        # Learned sparse adjacencies for visualization
    ├── sample_001_weighted_adj.pt
    └── sample_002_weighted_adj.pt

The comprehensive results file contains accuracy, F1 scores, ROC/PR AUC values with 95% confidence intervals, and confusion matrices. Sparse adjacency matrices can be used to generate heatmaps in the next step.
### Step 5: Visualization


Generate heatmaps overlaying learned edge weights on whole slide images (WSIs) to visualize model attention and analyze pruning effects.

### Usage

    python LENS_heatmap.py \
      --wsi-path /path/to/slide.svs \
      --patch-info-path /path/to/c_idx.txt \
      --pruned-adj-path /path/to/pruned_adj.pt \
      --original-adj-path /path/to/original_adj.pt \
      --output-dir heatmap_results

### Key Parameters

- `--wsi-path`: Path to whole slide image (.svs file)
- `--patch-info-path`: Path to patch coordinates file (c_idx.txt)  
- `--pruned-adj-path`: Path to pruned adjacency matrix from testing
- `--original-adj-path`: Path to original adjacency matrix (optional, for comparison)
- `--output-dir`: Output directory for visualization results

### Output

The script generates three types of visualizations:

#### Heatmap Overlays
    heatmap_results/pruned_heatmap/
    ├── sample_edge_weight_heatmap.png    # WSI with JET colormap overlay
    ├── sample_combined.png               # Original and heatmap side-by-side
    └── sample_weight_distribution.png    # Edge weight histogram

#### Comparison Analysis (if original provided)
    heatmap_results/comparison/
    ├── sample_weight_comparison.png      # Original vs pruned distributions
    └── sample_importance_diff.png        # Node importance changes

#### Statistics
Edge retention statistics and weight distributions are printed to console and saved in output files.

The heatmap uses JET colormap where red indicates high edge connectivity (important regions) and blue indicates low connectivity. This visualization reveals which tissue areas the model considers most important for classification.



## 📊 Results

LENS demonstrates robust discriminative power while utilizing only ~25% of the graph edges, indicating efficient extraction of relevant structural information.

<div align="center">
 <img width="1128" alt="Image" src="https://github.com/user-attachments/assets/3f0170c7-21c1-4048-94cb-1f5645237994" />
</div>

## 📚 Citation

If you use LENS in your research, please cite our paper:

```bibtex
@article{
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


# LENS-
# Lens_plus
