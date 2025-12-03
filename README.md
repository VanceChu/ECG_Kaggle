# ECG_Kaggle

## Environment Setup

We use a "clean" Conda environment strategy to ensure cross-platform compatibility.

- `environment.yml` installs all common dependencies
- PyTorch (and its CUDA stack) is installed **manually per machine**

### 1. Create Base Environment
The `environment.yml` contains all necessary libraries (pandas, numpy, scikit-learn, etc.) **except** PyTorch.

```bash
# Create the environment named 'ecg'
conda env create -f environment.yml

# Activate the environment
conda activate ecg
```
### 2. Install Pytorch manually
To avoid CUDA version conflicts, PyTorch must be installed manually based on your server's hardware.
```bash
# Example (For CUDA 12.x):
pip install torch torchvision torchaudio
```

### For Developers: Updating Dependencies
If you add new packages and want to update environment.yml, please use the provided script to strip hardware-specific dependencies:
```bash
# 1. Export raw environment
conda env export --no-builds > environment_raw.yml

# 2. Clean up (Remove system paths and PyTorch binaries)
python scripts/clean_env.py

# 3. Rename result to environment.yml
mv environment_clean.yml environment.yml
```