[English](README.md) | [简体中文](README_CN.md)

# ECG_Kaggle

## 环境配置

本项目使用“纯净” Conda 环境策略，以确保跨平台兼容性。

- `environment.yml` 用于安装所有通用的依赖库。
- PyTorch（及对应CUDA）需根据**具体服务器情况手动安装**。

### 1. 创建基础环境
`environment.yml` 包含所有必要的库（pandas, numpy, scikit-learn 等），**不包含PyTorch**。

```bash
# 创建名为 'ecg' 的环境
conda env create -f environment.yml

# 激活环境
conda activate ecg
```
### 2. 手动安装PyTorch

为了避免 CUDA 版本冲突，需要根据当前服务器的硬件与 CUDA 版本 手动安装 PyTorch。
```bash
# 示例（适用于 CUDA 12.x）
pip install torch torchvision torchaudio
```
### 开发者：更新依赖的流程

开发过程中新增依赖、更新 environment.yml 时，使用提供的脚本清理掉与硬件（PyTorch,CUDA）相关的部分：
```bash
# 1. 导出当前环境（原始版本）
conda env export --no-builds > environment_raw.yml

# 2. 清理导出的environment.yml中与硬件相关的依赖
python scripts/clean_env.py

# 3. 将硬件相关依赖清理后的文件重命名为 environment.yml
mv environment_clean.yml environment.yml
```

## 项目结构
```text
项目的目录结构如下：
ECG_KAGGLE/
├── docs/                           # 与 ECG 图像数字化相关的参考论文与资料
├── notebooks/                      # Kaggle 复现实验的 Notebook 与导出的 Python 脚本
│   ├── analysis/                   # Notebook 对应的 Markdown 分析与实验记录
│   ├── physio-v2-2-public.ipynb    # 复现 physio-v2-2-public 的主实验 Notebook
│   ├── physio-V2-2-public.py       # 上述 Notebook 导出的纯 Python 版本
│   ├── PhysioNet - Digitization of ECG Images.ipynb  # 复现 PhysioNet ECG 数字化方案的主 Notebook
│   └── PhysioNet - Digitization of ECG Images.py     # 上述 Notebook 导出的纯 Python 版本
├── src/                            # 核心源码模块与辅助函数
├── scripts/                        # 工具脚本库
│   └── clean_env.py                # 清理导出的environment.yml中与硬件相关的依赖
├── environment.yml                 # 用于复现的 Conda 环境配置
└── README.md                       
```
