# %% [markdown]
# Base on [https://www.kaggle.com/code/seshurajup/henkgck-submission-v4-credits-to-hengck](https://www.kaggle.com/code/seshurajup/henkgck-submission-v4-credits-to-hengck)
# 
# Main Changes:
# 
# 1. Seperate 3 stages to 3 files to avoid multiple models in GPU
# 2. Fix global seed to 0
# <!-- 3. Set `FLOAT_TYPE` from `float32` to `bfloat16` -->
# 4. Set `mv_to_pixel = 78.3` and `t0, t1 = 117, 2081`
# 5. Remove `filter_series_by_limits` cause I find it don't help LB
# 6. Many code refactoring

# %%
!pip uninstall -y tensorflow
!uv pip install --no-deps --system --no-index --find-links='/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/setup' 'connected-components-3d'

# %%
%%writefile constant.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import kagglehub
seed = 0
CUDA0 = "cuda:0"
deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(seed, disable_list=['cuda_block'])

import sys
sys.path.append('/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet')

import os
import traceback
from pathlib import Path
from shutil import copyfile
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

if_submit = os.getenv('KAGGLE_IS_COMPETITION_RERUN')

if if_submit:
    test_meta = Path("/kaggle/input/physionet-ecg-image-digitization/test.csv")
    test_dir = Path("/kaggle/input/physionet-ecg-image-digitization/test")
else:
    test_meta = Path("/kaggle/input/physio-test-fake-dataset/test_fake/test.csv")
    test_dir = Path("/kaggle/input/physio-test-fake-dataset/test_fake")

valid_df = pd.read_csv(test_meta)
valid_df['id'] = valid_df['id'].astype(str) 
valid_id = valid_df['id'].unique().tolist()

FLOAT_TYPE = torch.float32

global_dict = {
    "stage0_dir": "/kaggle/working/stage0",
    "stage1_dir": "/kaggle/working/stage1",
    "stage2_dir": "/kaggle/working/stage2",
}

# %%
%%writefile stage0.py

from constant import *

from stage0_model import Net as Stage0Net
from stage0_common import *

stage0_dir = Path(global_dict["stage0_dir"])
stage0_dir.mkdir(exist_ok=True)

stage0_net = Stage0Net(pretrained=False)
stage0_net = load_net(stage0_net, '/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage0-last.checkpoint.pth')
stage0_net.to(CUDA0)

for n, sample_id in enumerate(tqdm(valid_id)):
    path = test_dir / f'{sample_id}.png'
    output_path = stage0_dir / f'{sample_id}.png'
    image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
    batch = image_to_batch(image)

    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            output = stage0_net(batch)
        rotated, keypoint = output_to_predict(image, batch, output)
        normalised, _, _ = normalise_by_homography(rotated, keypoint)
        cv2.imwrite(output_path, cv2.cvtColor(normalised, cv2.COLOR_RGB2BGR))
    except:
        traceback.print_exc()
        copyfile(path, output_path)

# %%
%%writefile stage1.py

from constant import *

from stage1_model import Net as Stage1Net
from stage1_common import *

stage0_dir = Path(global_dict["stage0_dir"])
stage1_dir = Path(global_dict["stage1_dir"])
stage1_dir.mkdir(exist_ok=True)

stage1_net = Stage1Net(pretrained=False)
stage1_net = load_net(stage1_net, '/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage1-last.checkpoint.pth')
stage1_net.to(CUDA0)

for n, sample_id in enumerate(tqdm(valid_id)):
    path = stage0_dir / f'{sample_id}.png'
    output_path = stage1_dir / f'{sample_id}.png'
    image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
    batch = {'image': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)}

    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            output = stage1_net(batch)
        gridpoint_xy, _ = output_to_predict(image, batch, output)
        rectified = rectify_image(image, gridpoint_xy)
        cv2.imwrite(output_path, cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
    except:
        traceback.print_exc()
        copyfile(path, output_path)

# %%
%%writefile stage2.py

from constant import *

from stage2_model import Net as Stage2Net
from stage2_common import *

stage1_dir = Path(global_dict["stage1_dir"])
stage2_dir = Path(global_dict["stage2_dir"])
stage2_dir.mkdir(exist_ok=True)

stage2_net = Stage2Net(pretrained=False)
stage2_net = load_net(stage2_net, '/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage2-00005810.checkpoint.pth')
stage2_net.to(CUDA0)

x0, x1 = 0, 2176
y0, y1 = 0, 1696
zero_mv = [703.5, 987.5, 1271.5, 1531.5]
mv_to_pixel = 78.3
t0, t1 = 117, 2081

for n, sample_id in enumerate(tqdm(valid_id)):
    path = stage1_dir / f'{sample_id}.png'
    output_path = stage2_dir / f'{sample_id}.npy'
    image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
    length = valid_df[(valid_df['id']==sample_id) & (valid_df['lead']=='II')].iloc[0].number_of_rows
    image = image[y0:y1, x0:x1]
    batch = {'image': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)}
    
    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            output = stage2_net(batch)
        pixel = output['pixel'].float().data.cpu().numpy()[0]
        series_in_pixel = pixel_to_series(pixel[..., t0:t1], zero_mv, length)
        series = (np.array(zero_mv).reshape(4, 1) - series_in_pixel) / mv_to_pixel
        np.save(output_path, series)
    except:
        traceback.print_exc()
        series = np.zeros((4, length))
        np.save(output_path, series)

# %%
!python stage0.py
!python stage1.py
!python stage2.py

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_pred_gt(pred, gt):
    fig = make_subplots(rows=12, cols=1, subplot_titles=[f'Lead {i+1}' for i in range(12)])
    for i in range(12):
        fig.add_trace(go.Scatter(y=pred[:, i], mode='lines', name=f'Pred Lead {i+1}', line=dict(color='blue')), row=i+1, col=1)
        fig.add_trace(go.Scatter(y=gt[:, i], mode='lines', name=f'GT Lead {i+1}', line=dict(color='red')), row=i+1, col=1)
    fig.update_layout(height=1200, showlegend=False)
    fig.show(renderer='iframe')

def expand_4_to_12(pred4):
    pred12 = np.zeros((pred4.shape[0], 12))
    quat = pred4.shape[0] // 4

    pred12[:quat, 0] = pred4[:quat, 0]
    pred12[:, 1] = pred4[:, 3]
    pred12[:quat, 2] = pred4[:quat, 2]
    pred12[quat:2*quat, 3] = pred4[quat:2*quat, 0]
    pred12[quat:2*quat, 4] = pred4[quat:2*quat, 1]
    pred12[quat:2*quat, 5] = pred4[quat:2*quat, 2]
    pred12[2*quat:3*quat, 6] = pred4[2*quat:3*quat, 0]
    pred12[2*quat:3*quat, 7] = pred4[2*quat:3*quat, 1]
    pred12[2*quat:3*quat, 8] = pred4[2*quat:3*quat, 2]
    pred12[3*quat:4*quat, 9] = pred4[3*quat:4*quat, 0]
    pred12[3*quat:4*quat, 10] = pred4[3*quat:4*quat, 1]
    pred12[3*quat:4*quat, 11] = pred4[3*quat:4*quat, 2]
    
    return pred12

def series_dict(series):
    series_by_lead = dict()
    for l in range(3):
        lead_names = [
            ['I',   'aVR', 'V1', 'V4'],
            ['II',  'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
        ][l]
        split = np.array_split(series[l], 4)
        for (k, s) in zip(lead_names, split):
            series_by_lead[k] = s
    series_by_lead['II'] = series[3]
    return series_by_lead

# %%
from constant import *

stage2_dir = Path(global_dict["stage2_dir"])

submit_df = list()
gb = valid_df.groupby('id')

show = True

for rec_idx, (sample_id, df) in enumerate(tqdm(gb)):
    series = np.load(stage2_dir / f'{sample_id}.npy')

    if not if_submit and show:
        pred = np.transpose(series, axes=(1, 0))
        pred = expand_4_to_12(pred)
        gt = pd.read_csv("/kaggle/input/physio-test-fake-dataset/7663343_inp1.csv").fillna(0).values
        show_pred_gt(pred, gt)
        show = False
    
    series_by_lead = series_dict(series)

    for _, d in df.iterrows():
        s = series_by_lead[d.lead]
        if len(s) != d.number_of_rows:
            x_old = np.linspace(0.0, 1.0, len(s))
            x_new = np.linspace(0.0, 1.0, d.number_of_rows)
            s = np.interp(x_new, x_old, s)
        row_id = [f'{sample_id}_{t}_{d.lead}' for t in range(d.number_of_rows)]
        this_df = pd.DataFrame({
            'id': row_id,
            'value': s,
        })
        submit_df.append(this_df)

submit_df = pd.concat(submit_df, axis=0, ignore_index=True, sort=False, copy=False)
submit_df.to_csv('submission.csv', index=False)

# %%
sub = pd.read_csv('submission.csv')
print(len(sub))
sub.head(30)


