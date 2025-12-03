[English](README.md) | [简体中文](README_CN.md)
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

## Project Structure

The directory structure of the project is organized as follows:

```text
ECG_KAGGLE/
├── docs/                           # Reference papers and literature related to ECG digitization
├── notebooks/                      # Kaggle reproduction notebooks and exported Python scripts
│   ├── analysis/                   # Markdown analysis and experiment notes for the notebooks
│   ├── physio-v2-2-public.ipynb    # Main notebook: reproduction of physio-v2-2-public
│   ├── physio-V2-2-public.py       # Pure Python version exported from the notebook above
│   ├── PhysioNet - Digitization of ECG Images.ipynb  # Main notebook: PhysioNet ECG digitization
│   └── PhysioNet - Digitization of ECG Images.py     # Pure Python version exported from the notebook
├── scripts/                        # Utility scripts (e.g., environment cleaning)
├── src/                            # Core source code modules and helper functions
├── environment.yml                 # Reproducible Conda environment configuration
└── README.md                       # Project description and usage instructions
