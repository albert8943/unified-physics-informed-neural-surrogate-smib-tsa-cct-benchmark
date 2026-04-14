# Training Module

This directory contains training scripts for different PINN tasks. Each script provides a complete training pipeline with configuration management, logging, and checkpointing.

## 📑 Table of Contents

- [Overview](#overview)
- [Module Files](#module-files)
  - [train_trajectory.py](#train_trajectorypy)
  - [train_parameters.py](#train_parameterspy)
  - [train_cct.py](#train_cctpy)
- [Configuration Files](#configuration-files)
  - [configs/training/trajectory_config.yaml](#configstrainingtrajectory_configyaml)
  - [configs/training/parameter_config.yaml](#configstrainingparameter_configyaml)
  - [configs/training/cct_config.yaml](#configstrainingcct_configyaml)
- [Cloud GPU Training (Google Colab / Kaggle)](#cloud-gpu-training-google-colab--kaggle)
  - [Google Colab Setup](#google-colab-setup)
  - [Kaggle Setup](#kaggle-setup)
  - [Cloud Training Best Practices](#cloud-training-best-practices)
  - [Example Colab Notebook Structure](#example-colab-notebook-structure)
  - [Troubleshooting Cloud Training](#troubleshooting-cloud-training)
  - [Alternative: Local GPU Training](#alternative-local-gpu-training)
- [Training Workflow](#training-workflow)
  - [Standard Training Process](#standard-training-process)
  - [Example: Training Trajectory Prediction Model](#example-training-trajectory-prediction-model)
- [Command-Line Arguments](#command-line-arguments)
- [Checkpointing](#checkpointing)
- [Logging](#logging)
- [Visualization During Training](#visualization-during-training)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
- [Notes](#notes)

---

## Overview

The `training/` module includes:
- **Task-specific training scripts** - Separate scripts for each PINN task
- **Configuration files** - YAML configuration files for training hyperparameters
- **Training pipelines** - Complete workflows from data loading to model saving

## Module Files

### `train_trajectory.py`
**Purpose**: Train PINN model for trajectory prediction (δ, ω).

**Features**:
- Loads trajectory prediction training data
- Initializes `TrajectoryPredictionPINN` model
- Trains with physics-informed loss
- Evaluates on validation set
- Saves model checkpoints and training logs
- Generates trajectory plots

**Usage**:
```bash
python training/train_trajectory.py --config configs/training/trajectory_config.yaml
```

**Configuration** (`configs/training/trajectory_config.yaml`):
- Model architecture (hidden layers, activation)
- Training hyperparameters (learning rate, batch size, epochs)
- Loss weights (lambda_data, lambda_physics, lambda_ic)
- Data paths
- Output paths

**Outputs**:
- Trained model checkpoint: `outputs/models/trajectory_model_*.pth`
- Training logs: `outputs/logs/trajectory_training_*.log`
- Loss curves: `outputs/figures/trajectory_loss_curves.png`
- Sample predictions: `outputs/figures/trajectory_predictions.png`

---

### `train_parameters.py`
**Purpose**: Train PINN model for parameter estimation (H, D).

**Features**:
- Loads parameter estimation training data (observed trajectories)
- Initializes `ParameterEstimationPINN` model
- Trains to estimate H and D from trajectories
- Evaluates parameter estimation accuracy
- Saves model checkpoints

**Usage**:
```bash
python training/train_parameters.py --config configs/training/parameter_config.yaml
```

**Configuration** (`configs/training/parameter_config.yaml`):
- Model architecture (LSTM vs fully connected)
- Sequence length for time series input
- Training hyperparameters
- Parameter ranges for normalization

**Outputs**:
- Trained model checkpoint: `outputs/models/parameter_model_*.pth`
- Training logs: `outputs/logs/parameter_training_*.log`
- Parameter estimation metrics

---

### `train_cct.py`
**Purpose**: Train PINN model for CCT estimation (deprecated approach).

**Note**: This script uses the deprecated `CCTEstimationPINN` class. The recommended approach is to:
1. Train a trajectory prediction model using `train_trajectory.py`
2. Use binary search with the trained model (see `utils.cct_binary_search`)

**Status**: Kept for reference but not recommended for new implementations.

**Usage** (if needed):
```bash
python training/train_cct.py --config configs/training/cct_config.yaml
```

---

## Configuration Files

### `configs/trajectory_config.yaml`
Configuration for trajectory prediction training:
```yaml
model:
  input_dim: 10
  hidden_dims: [64, 64, 64, 64]
  activation: "tanh"
  use_residual: true

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 1000
  device: "cuda"

loss:
  lambda_data: 1.0
  lambda_physics: 0.1
  lambda_ic: 10.0
  lambda_boundary: 1.0

data:
  train_path: "data/processed/trajectory_train.csv"
  val_path: "data/processed/trajectory_val.csv"
  test_path: "data/processed/trajectory_test.csv"

output:
  model_dir: "outputs/models"
  log_dir: "outputs/logs"
  figure_dir: "outputs/figures"
```

### `configs/parameter_config.yaml`
Configuration for parameter estimation training:
```yaml
model:
  input_dim: 2
  sequence_length: 100
  hidden_dims: [128, 128, 64]
  use_lstm: true

training:
  batch_size: 16
  learning_rate: 0.0005
  epochs: 500
  device: "cuda"

data:
  train_path: "data/processed/parameter_train.csv"
  val_path: "data/processed/parameter_val.csv"
```

### `configs/cct_config.yaml`
Configuration for CCT estimation training (deprecated):
- Similar structure to trajectory config
- Note: Use binary search approach instead

---

## Cloud GPU Training (Google Colab / Kaggle)

The framework supports cloud GPU training on Google Colab and Kaggle for faster training without local GPU hardware. Here are setup instructions for both platforms:

### Google Colab Setup

1. **Create a New Colab Notebook**:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Create a new notebook
   - Enable GPU: Runtime → Change runtime type → Hardware accelerator: GPU (T4 recommended)

2. **Install Dependencies**:
   ```python
   # Install ANDES and dependencies (Colab already has most Python packages)
   !pip install andes
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Install PINN dependencies
   !pip install scikit-learn pyyaml tqdm matplotlib

   # Install scipy if needed (usually pre-installed)
   !pip install scipy
   ```

3. **Mount Google Drive** (optional, for data/model storage):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   # Set working directory
   import os
   os.chdir('/content/drive/MyDrive/PINN_project_using_andes')
   ```

4. **Upload Project Files**:
   - Option A: Clone from GitHub
     ```python
     !git clone https://github.com/yourusername/PINN_project_using_andes.git
     %cd PINN_project_using_andes
     ```
   - Option B: Upload zip file and extract
     ```python
     from google.colab import files
     uploaded = files.upload()  # Upload project zip
     !unzip project.zip
     %cd PINN_project_using_andes
     ```

5. **Upload Data** (if pre-generated):
   ```python
   # Upload data directory
   from google.colab import files
   # Or use Drive if data is large
   !cp -r /content/drive/MyDrive/data /content/PINN_project_using_andes/
   ```

6. **Train Model**:
   ```python
   # Train trajectory prediction model on GPU
   !python training/train_trajectory.py \
       --config configs/training/trajectory_config.yaml \
       --data_dir data/preprocessed \
       --output_dir outputs/trajectory \
       --device cuda
   ```

7. **Download Results**:
   ```python
   # Download trained models and results
   from google.colab import files
   files.download('outputs/trajectory/final_model.pt')

   # Or save to Drive
   !cp -r outputs /content/drive/MyDrive/PINN_results/
   ```

### Kaggle Setup

1. **Create a New Kaggle Notebook**:
   - Go to [Kaggle](https://www.kaggle.com/)
   - Create a new notebook
   - Enable GPU: Settings → Accelerator → GPU T4 x2 (or P100)

2. **Add Data Sources**:
   - Upload your project files as a dataset, or
   - Add data from existing datasets
   - Mount datasets in the notebook

3. **Install Dependencies**:
   ```python
   # Kaggle notebooks come with many packages pre-installed
   # Install ANDES
   !pip install andes

   # Install additional dependencies if needed
   !pip install pyyaml
   ```

4. **Set Up Project Structure**:
   ```python
   import os
   os.makedirs('data', exist_ok=True)
   os.makedirs('outputs', exist_ok=True)

   # Copy project files (if uploaded as dataset)
   !cp -r ../input/pinn-project/* .
   ```

5. **Train Model**:
   ```python
   # Train on GPU
   !python training/train_trajectory.py \
       --config configs/training/trajectory_config.yaml \
       --data_dir data/preprocessed \
       --output_dir outputs/trajectory \
       --device cuda
   ```

6. **Save Outputs**:
   - Kaggle automatically saves notebook outputs
   - Download models/results from the notebook outputs section

### Cloud Training Best Practices

1. **Data Management**:
   - For large datasets, use Google Drive or Kaggle Datasets
   - Pre-generate data locally and upload (ANDES simulation can be slow on cloud)
   - Use compressed formats (zip, tar.gz) for faster uploads

2. **Checkpointing**:
   - Save checkpoints frequently (every 10-20 epochs)
   - Use cloud storage (Drive, Kaggle outputs) for persistence
   - Implement automatic checkpoint saving in training scripts

3. **Resource Management**:
   - Monitor GPU memory usage
   - Adjust batch size if out-of-memory errors occur
   - Use mixed precision training for faster training and lower memory
     ```python
     from torch.cuda.amp import autocast, GradScaler
     scaler = GradScaler()

     with autocast():
         loss = model.compute_loss(...)
     scaler.scale(loss).backward()
     scaler.step(optimizer)
     scaler.update()
     ```

4. **Time Limits**:
   - **Google Colab**: Free tier has 12-hour session limits
   - **Kaggle**: Free tier has 9-hour session limits (30 hours/week)
   - Plan training accordingly or use paid tiers for longer sessions

5. **Data Generation on Cloud**:
   - ANDES can run on Colab/Kaggle but may be slower
   - Recommended: Generate data locally, upload to cloud
   - For cloud data generation, use smaller parameter sweeps initially

### Example Colab Notebook Structure

```python
# Cell 1: Install dependencies
!pip install andes torch scikit-learn pyyaml tqdm matplotlib scipy

# Cell 2: Setup
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Cell 3: Mount Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

# Cell 4: Clone or upload project
!git clone https://github.com/yourusername/PINN_project_using_andes.git
%cd PINN_project_using_andes

# Cell 5: Upload data (or generate)
# Option: Upload pre-generated data
# Option: Generate data (slower)
from data_generation import generate_trajectory_data
df = generate_trajectory_data(
    case_file="smib/SMIB.json",
    output_dir="data/trajectory",
    H_range=(2.0, 10.0, 5),
    D_range=(0.5, 3.0, 5),
    fault_clearing_times=[0.15, 0.18, 0.20, 0.22, 0.25]
)

# Cell 6: Preprocess data
from data_generation import preprocess_data
result = preprocess_data(
    data=df,
    normalize=True,
    split=True,
    output_dir="data/preprocessed"
)

# Cell 7: Train model
!python training/train_trajectory.py \
    --config configs/training/trajectory_config.yaml \
    --data_dir data/preprocessed \
    --output_dir outputs/trajectory \
    --device cuda

# Cell 8: Download results
from google.colab import files
files.download('outputs/trajectory/final_model.pt')
```

### Troubleshooting Cloud Training

1. **Out of Memory Errors**:
   - Reduce batch size in config file
   - Use gradient accumulation
   - Enable mixed precision training

2. **Session Timeouts**:
   - Save checkpoints frequently
   - Resume training from last checkpoint
   - Use paid tiers for longer sessions

3. **Slow Data Loading**:
   - Pre-process data locally
   - Use compressed formats
   - Cache data in memory if possible

4. **ANDES Installation Issues**:
   - Colab/Kaggle may have limited system dependencies
   - Consider pre-generating data locally
   - Use conda environments if needed (Colab Pro)

### Alternative: Local GPU Training

If you have a local GPU, you can train locally for better control:

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Train on GPU
python training/train_trajectory.py \
    --config configs/training/trajectory_config.yaml \
    --data_dir data/preprocessed \
    --output_dir outputs/trajectory \
    --device cuda
```

## Training Workflow

### Standard Training Process

1. **Prepare Data**: Generate training data using `data_generation` module
2. **Configure Training**: Edit YAML config files or use command-line arguments
3. **Run Training**: Execute training script
4. **Monitor Progress**: Check logs and loss curves
5. **Evaluate Model**: Use evaluation scripts in `evaluation/` module

### Example: Training Trajectory Prediction Model

```bash
# 1. Generate training data (if not already done)
python -m data_generation.parameter_sweep generate_trajectory_data

# 2. Train model
python training/train_trajectory.py \
    --config configs/training/trajectory_config.yaml \
    --epochs 1000 \
    --batch-size 32

# 3. Monitor training
tail -f outputs/logs/trajectory_training_*.log

# 4. Evaluate trained model
python evaluation/comprehensive_evaluation.py \
    --model outputs/models/trajectory_model_*.pth \
    --task trajectory
```

---

## Command-Line Arguments

All training scripts support common arguments:

```bash
python training/train_trajectory.py \
    --config <config_file> \          # YAML config file
    --epochs <n> \                    # Number of training epochs
    --batch-size <n> \               # Batch size
    --learning-rate <lr> \            # Learning rate
    --device <device> \               # "cpu" or "cuda"
    --resume <checkpoint> \           # Resume from checkpoint
    --output-dir <dir>                # Output directory
```

---

## Checkpointing

Training scripts automatically save checkpoints:
- **Best model**: Saved when validation loss improves
- **Periodic checkpoints**: Saved every N epochs (configurable)
- **Final model**: Saved at end of training

Checkpoint format:
```python
{
    "epoch": epoch_number,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss_value,
    "metrics": metrics_dict,
    "config": config_dict
}
```

**Resuming Training**:
```bash
python training/train_trajectory.py \
    --config configs/training/trajectory_config.yaml \
    --resume outputs/models/trajectory_model_checkpoint_epoch_500.pth
```

---

## Logging

Training logs include:
- Training loss (total and components: data, physics, IC)
- Validation loss and metrics
- Learning rate schedule
- Training time per epoch
- GPU memory usage (if using CUDA)

Log files are saved to `outputs/logs/` with timestamps.

---

## Visualization During Training

Training scripts generate:
- **Loss curves**: Training and validation loss over epochs
- **Component losses**: Breakdown of data, physics, and IC losses
- **Sample predictions**: Trajectory plots for validation samples
- **Learning curves**: Metrics over training epochs

Figures are saved to `outputs/figures/` directory.

---

## Best Practices

1. **Start with small models**: Test with smaller networks before scaling up
2. **Monitor loss components**: Ensure physics loss is decreasing (not just data loss)
3. **Use validation set**: Regular validation prevents overfitting
4. **Save checkpoints**: Enable resuming and model comparison
5. **Tune loss weights**: Balance between data fit and physics constraints
6. **Use appropriate batch sizes**: Larger batches for stable gradients, smaller for memory constraints

---

## Troubleshooting

### Common Issues

1. **Loss not decreasing**:
   - Check learning rate (may be too high/low)
   - Verify data normalization
   - Check loss weight balance

2. **Physics loss too high**:
   - Increase `lambda_physics` weight
   - Add more collocation points
   - Check physics equation implementation

3. **Out of memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Reduce model size

4. **Training too slow**:
   - Use GPU if available
   - Reduce number of collocation points
   - Use mixed precision training

---

## Notes

- All training scripts use PyTorch and support both CPU and GPU
- Configuration files use YAML format for easy modification
- Training progress can be monitored via logs and TensorBoard (if configured)
- Models are saved in PyTorch `.pth` format
- For CCT estimation, use binary search with trained trajectory model instead of `train_cct.py`

---

**[⬆ Back to Top](#training-module)**
