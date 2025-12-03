# %%
try:
    import cc3d
except:
    !pip install connected-components-3d --no-index --find-links=file:///kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/setup/

import cc3d
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add hengck23's modified library (by kami1976) to path
sys.path.append('/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet')

from stage0_common import image_to_batch, output_to_predict, normalise_by_homography, load_net
from stage0_model import Net as Stage0Net
from stage1_common import output_to_predict as stage1_output_to_predict, rectify_image
from stage1_model import Net as Stage1Net
from stage2_common import pixel_to_series, filter_series_by_limits
from stage2_model import Net as Stage2Net

# %%
# Scoring functions from PhysioNet ECG Digitization competition

from typing import Tuple

import numpy as np
import pandas as pd

import scipy.optimize
import scipy.signal


LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
MAX_TIME_SHIFT = 0.2
PERFECT_SCORE = 384


class ParticipantVisibleError(Exception):
    pass


def compute_power(label: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    if label.ndim != 1 or prediction.ndim != 1:
        raise ParticipantVisibleError('Inputs must be 1-dimensional arrays.')
    finite_mask = np.isfinite(prediction)
    if not np.any(finite_mask):
        raise ParticipantVisibleError("The 'prediction' array contains no finite values (all NaN or inf).")

    prediction[~np.isfinite(prediction)] = 0
    noise = label - prediction
    p_signal = np.sum(label**2)
    p_noise = np.sum(noise**2)
    return p_signal, p_noise


def compute_snr(signal: float, noise: float) -> float:
    if noise == 0:
        # Perfect reconstruction
        snr = PERFECT_SCORE
    elif signal == 0:
        snr = 0
    else:
        snr = min((signal / noise), PERFECT_SCORE)
    return snr


def align_signals(label: np.ndarray, pred: np.ndarray, max_shift: float = float('inf')) -> np.ndarray:
    if np.any(~np.isfinite(label)):
        raise ParticipantVisibleError('values in label should all be finite')
    if np.sum(np.isfinite(pred)) == 0:
        raise ParticipantVisibleError('prediction can not all be infinite')

    # Initialize the reference and digitized signals.
    label_arr = np.asarray(label, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)

    label_mean = np.mean(label_arr)
    pred_mean = np.mean(pred_arr)

    label_arr_centered = label_arr - label_mean
    pred_arr_centered = pred_arr - pred_mean

    # Compute the correlation between the reference and digitized signals and locate the maximum correlation.
    correlation = scipy.signal.correlate(label_arr_centered, pred_arr_centered, mode='full')

    n_label = np.size(label_arr)
    n_pred = np.size(pred_arr)

    lags = scipy.signal.correlation_lags(n_label, n_pred, mode='full')
    valid_lags_mask = (lags >= -max_shift) & (lags <= max_shift)

    max_correlation = np.nanmax(correlation[valid_lags_mask])
    all_max_indices = np.flatnonzero(correlation == max_correlation)
    best_idx = min(all_max_indices, key=lambda i: abs(lags[i]))
    time_shift = lags[best_idx]
    start_padding_len = max(time_shift, 0)
    pred_slice_start = max(-time_shift, 0)
    pred_slice_end = min(n_label - time_shift, n_pred)
    end_padding_len = max(n_label - n_pred - time_shift, 0)
    aligned_pred = np.concatenate((np.full(start_padding_len, np.nan), pred_arr[pred_slice_start:pred_slice_end], np.full(end_padding_len, np.nan)))

    def objective_func(v_shift):
        return np.nansum((label_arr - (aligned_pred - v_shift)) ** 2)

    if np.any(np.isfinite(label_arr) & np.isfinite(aligned_pred)):
        results = scipy.optimize.minimize_scalar(objective_func, method='Brent')
        vertical_shift = results.x
        aligned_pred -= vertical_shift
    return aligned_pred


def _calculate_image_score(group: pd.DataFrame) -> float:
    """Helper function to calculate the total SNR score for a single image group."""

    unique_fs_values = group['fs'].unique()
    if len(unique_fs_values) != 1:
        raise ParticipantVisibleError('Sampling frequency should be consistent across each ecg')
    sampling_frequency = unique_fs_values[0]
    if sampling_frequency != int(len(group[group['lead'] == 'II']) / 10):
        raise ParticipantVisibleError('The sequence_length should be sampling frequency * 10s')
    sum_signal = 0
    sum_noise = 0
    for lead in LEADS:
        sub = group[group['lead'] == lead]
        label = sub['value_true'].values
        pred = sub['value_pred'].values

        aligned_pred = align_signals(label, pred, int(sampling_frequency * MAX_TIME_SHIFT))
        p_signal, p_noise = compute_power(label, aligned_pred)
        sum_signal += p_signal
        sum_noise += p_noise
    return compute_snr(sum_signal, sum_noise)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute the mean Signal-to-Noise Ratio (SNR) across multiple ECG leads and images for the PhysioNet 2025 competition.
    The final score is the average of the sum of SNRs over different lines, averaged over all unique images.
    Args:
        solution: DataFrame with ground truth values. Expected columns: 'id' and one for each lead.
        submission: DataFrame with predicted values. Expected columns: 'id' and one for each lead.
        row_id_column_name: The name of the unique identifier column, typically 'id'.
    Returns:
        The final competition score.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> row_id_column_name = "id"
    >>> solution = pd.DataFrame({'id': ['343_0_I', '343_1_I', '343_2_I', '343_0_III', '343_1_III','343_2_III','343_0_aVR', '343_1_aVR','343_2_aVR',\
    '343_0_aVL', '343_1_aVL', '343_2_aVL', '343_0_aVF', '343_1_aVF','343_2_aVF','343_0_V1', '343_1_V1', '343_2_V1','343_0_V2', '343_1_V2','343_2_V2',\
    '343_0_V3', '343_1_V3', '343_2_V3','343_0_V4', '343_1_V4', '343_2_V4', '343_0_V5', '343_1_V5','343_2_V5','343_0_V6', '343_1_V6','343_2_V6',\
    '343_0_II', '343_1_II','343_2_II', '343_3_II', '343_4_II', '343_5_II','343_6_II', '343_7_II','343_8_II','343_9_II','343_10_II','343_11_II'],\
    'fs': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\
    'value':[0.1,0.3,0.4,0.6,0.6,0.4,0.2,0.3,0.4,0.5,0.2,0.7,0.2,0.3,0.4,0.8,0.6,0.7, 0.2,0.3,-0.1,0.5,0.6,0.7,0.2,0.9,0.4,0.5,0.6,0.7,0.1,0.3,0.4,\
    0.6,0.6,0.4,0.2,0.3,0.4,0.5,0.2,0.7,0.2,0.3,0.4]})
    >>> submission = solution.copy()
    >>> round(score(solution, submission, row_id_column_name), 4)
    25.8433
    >>> submission.loc[0, 'value'] = 0.9 # Introduce some noise
    >>> round(score(solution, submission, row_id_column_name), 4)
    13.6291
    >>> submission.loc[4, 'value'] = 0.3 # Introduce some noise
    >>> round(score(solution, submission, row_id_column_name), 4)
    13.0576

    >>> solution = pd.DataFrame({'id': ['343_0_I', '343_1_I', '343_2_I', '343_0_III', '343_1_III','343_2_III','343_0_aVR', '343_1_aVR','343_2_aVR',\
    '343_0_aVL', '343_1_aVL', '343_2_aVL', '343_0_aVF', '343_1_aVF','343_2_aVF','343_0_V1', '343_1_V1', '343_2_V1','343_0_V2', '343_1_V2','343_2_V2',\
    '343_0_V3', '343_1_V3', '343_2_V3','343_0_V4', '343_1_V4', '343_2_V4', '343_0_V5', '343_1_V5','343_2_V5','343_0_V6', '343_1_V6','343_2_V6',\
    '343_0_II', '343_1_II','343_2_II', '343_3_II', '343_4_II', '343_5_II','343_6_II', '343_7_II','343_8_II','343_9_II','343_10_II','343_11_II'],\
    'fs': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\
    'value':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]})
    >>> round(score(solution, submission, row_id_column_name), 4)
    -384
    >>> submission = solution.copy()
    >>> round(score(solution, submission, row_id_column_name), 4)
    25.8433

    >>> # test alignment
    >>> label = np.array([0, 1, 2, 1, 0])
    >>> pred = np.array([0, 1, 2, 1, 0])
    >>> aligned = align_signals(label, pred)
    >>> expected_array = np.array([0, 1, 2, 1, 0])
    >>> np.allclose(aligned, expected_array, equal_nan=True)
    True

    >>> # Test 2: Vertical shift (DC offset) should be removed
    >>> label = np.array([0, 1, 2, 1, 0])
    >>> pred = np.array([10, 11, 12, 11, 10])
    >>> aligned = align_signals(label, pred)
    >>> expected_array = np.array([0, 1, 2, 1, 0])
    >>> np.allclose(aligned, expected_array, equal_nan=True)
    True

    >>> # Test 3: Time shift should be corrected
    >>> label = np.array([0, 0, 1, 2, 1, 0., 0.])
    >>> pred = np.array([1, 2, 1, 0, 0, 0, 0])
    >>> aligned = align_signals(label, pred)
    >>> expected_array = np.array([np.nan, np.nan, 1, 2, 1, 0, 0])
    >>> np.allclose(aligned, expected_array, equal_nan=True)
    True

    >>> # Test 4: max_shift constraint prevents optimal alignment
    >>> label = np.array([0, 0, 0, 0, 1, 2, 1]) # Peak is far
    >>> pred = np.array([1, 2, 1, 0, 0, 0, 0])
    >>> aligned = align_signals(label, pred, max_shift=10)
    >>> expected_array = np.array([ np.nan, np.nan, np.nan, np.nan, 1, 2, 1])
    >>> np.allclose(aligned, expected_array, equal_nan=True)
    True

    """
    for df in [solution, submission]:
        if row_id_column_name not in df.columns:
            raise ParticipantVisibleError(f"'{row_id_column_name}' column not found in DataFrame.")
        if df['value'].isna().any():
            raise ParticipantVisibleError('NaN exists in solution/submission')
        if not np.isfinite(df['value']).all():
            raise ParticipantVisibleError('Infinity exists in solution/submission')

    submission = submission[['id', 'value']]
    merged_df = pd.merge(solution, submission, on=row_id_column_name, suffixes=('_true', '_pred'))
    merged_df['image_id'] = merged_df[row_id_column_name].str.split('_').str[0]
    merged_df['row_id'] = merged_df[row_id_column_name].str.split('_').str[1].astype('int64')
    merged_df['lead'] = merged_df[row_id_column_name].str.split('_').str[2]
    merged_df.sort_values(by=['image_id', 'row_id', 'lead'], inplace=True)
    image_scores = merged_df.groupby('image_id').apply(_calculate_image_score, include_groups=False)
    return max(float(10 * np.log10(image_scores.mean())), -PERFECT_SCORE)

# %%
# ECG lead structure for 4-row format
# Rows 0-2 each contain 4 concatenated short leads (~2.5s each)
# Row 3 contains the full 10s Lead II rhythm strip
LEAD_NAMES_BY_ROW = [
    ['I', 'aVR', 'V1', 'V4'],        # Row 0
    ['II_short', 'aVL', 'V2', 'V5'], # Row 1 (II_short is the short segment)
    ['III', 'aVF', 'V3', 'V6'],      # Row 2
]

# Type IDs for validation
TYPE_IDS = ['0001', '0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012']


def interpolate_signal_to_length(signal, target_length):
    """
    Interpolate signal to match target length.

    Length correction by interpolation (from kami1976)
    https://www.kaggle.com/code/kami1976/physionet-digitization-of-ecg-images-v22

    Args:
        signal: Input signal array
        target_length: Desired output length

    Returns:
        Interpolated signal of length target_length
    """
    if len(signal) == target_length:
        return signal

    x_old = np.linspace(0.0, 1.0, len(signal), endpoint=False)
    x_new = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(x_new, x_old, signal)


def extract_leads_from_series(series):
    """
    Extract individual lead signals from 4-row series format.

    Args:
        series: (4, L) array from stage2 output
            Row 0: I, aVR, V1, V4 concatenated
            Row 1: II, aVL, V2, V5 concatenated
            Row 2: III, aVF, V3, V6 concatenated
            Row 3: Full Lead II rhythm strip

    Returns:
        dict: Mapping of lead_name -> signal array for all 12 leads
    """
    L = series.shape[1]
    predicted_leads = {}

    # Extract leads using array_split for rows 0-2
    for row_idx in range(3):
        split = np.array_split(series[row_idx], 4)

        for lead_name, signal in zip(LEAD_NAMES_BY_ROW[row_idx], split):
            predicted_leads[lead_name] = signal

    # Row 3: Full Lead II rhythm strip
    # ENSEMBLE: Average Row 1 Position 0 (II_short) with Row 3 first quarter
    # Both represent the same electrical signal but from different image locations
    # Averaging provides robustness against localized image artifacts
    lead_ii = series[3].copy()
    ii_short = predicted_leads['II_short']  # From Row 1, Position 0

    # Average the first quarter (use actual length of ii_short for robustness)
    quarter_len = len(ii_short)
    lead_ii[:quarter_len] = (ii_short + lead_ii[:quarter_len]) / 2.0

    predicted_leads['II'] = lead_ii

    return predicted_leads


# [DL] This function copied from https://www.kaggle.com/code/hengck23/demo-submission
# (make_test_fake_df()). Simulates test.csv metadata from train.csv by reading
# ground truth CSVs.
def test_from_train_df(data_dir='/kaggle/input/physionet-ecg-image-digitization'):
    valid_df = pd.read_csv(f'{data_dir}/train.csv')
    valid_df['id'] = valid_df['id'].astype(str)
    fake_test_df=[]
    for i,d in valid_df.iterrows():
        #if i==4: break
        image_id = d['id']

        truth_df = pd.read_csv(f'{data_dir}/train/{image_id}/{image_id}.csv')
        non_nan_count = truth_df.count()
        #print(i,image_id,non_nan_count)
        #print(non_nan_count.index)

        #lead	fs	number_of_rows
        this_df = pd.DataFrame({
            'id':image_id ,
            'lead':non_nan_count.index,
            'fs': d['fs'],
            'number_of_rows':non_nan_count.values
        })
        fake_test_df.append(this_df)
        if i==0: print(this_df)
    fake_test_df = pd.concat(fake_test_df)
    return fake_test_df


def validation_subset(image_ids, type_ids,
                      data_dir='/kaggle/input/physionet-ecg-image-digitization'):
    """
    Iterator that yields image filenames and lead records for validation.

    Args:
        image_ids: List of image IDs to process (e.g., ['7663343', '7663344'])
        type_ids: List of type IDs to process (e.g., ['0001', '0003'])
        data_dir: Path to PhysioNet data directory

    Yields:
        tuple: (filename, lead_records)
            filename: Path to image file
            lead_records: DataFrame with lead specifications for this image
                (columns: id, lead, fs, number_of_rows)
    """
    # Load train.csv to get valid image IDs
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    train_df['id'] = train_df['id'].astype(str)
    valid_image_ids = set(train_df['id'].values)

    # Get lead records using test_from_train_df
    test_df = test_from_train_df(data_dir)

    # Filter requested image_ids to only those in train.csv
    filtered_image_ids = [img_id for img_id in image_ids if img_id in valid_image_ids]

    # Iterate through combinations
    for image_id in filtered_image_ids:
        # Get lead records for this image_id
        lead_records = test_df[test_df['id'] == image_id].copy()

        for type_id in type_ids:
            filename = f'{data_dir}/train/{image_id}/{image_id}-{type_id}.png'
            yield filename, lead_records


def plot_stage0_result(original, normalised, title='Stage 0 Result', show=True):
    """
    Plot stage0 input and output side by side.

    Args:
        original: Original input image (RGB)
        normalised: Normalized output image (RGB)
        title: Plot title
        show: If True, display plot on screen. If False, only prepare for saving.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(normalised)
    axes[1].set_title('Normalized Image', fontsize=14)
    axes[1].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if show:
        plt.show()


def plot_stage1_result(normalised, rectified, title='Stage 1 Result', show=True):
    """
    Plot stage1 input and output side by side.

    Args:
        normalised: Normalized input image from stage0 (RGB)
        rectified: Rectified output image (RGB)
        title: Plot title
        show: If True, display plot on screen. If False, only prepare for saving.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(normalised)
    axes[0].set_title('Normalized Image (from Stage 0)', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(rectified)
    axes[1].set_title('Rectified Image', fontsize=14)
    axes[1].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if show:
        plt.show()


def plot_stage2_result(predicted_leads, title='Stage 2 Result', ground_truth=None, overall_snr_db=None, lead_snrs=None, show=True):
    """
    Plot stage2 extracted signals as 12 leads (3x4) plus full Lead II.

    Args:
        predicted_leads: dict mapping lead_name -> signal array for all 12 leads
        title: Plot title
        ground_truth: Optional pd.DataFrame with ground truth signals
        overall_snr_db: Optional overall image SNR in dB to display in main title
        lead_snrs: Optional dict mapping lead_name -> SNR in dB (pre-calculated)
        show: If True, display plot on screen. If False, only prepare for saving.
    """

    # 3x4 grid for 12 leads
    fig1, axes = plt.subplots(3, 4, figsize=(16, 9))
    title_text = f'{title}'
    if overall_snr_db is not None:
        title_text += f' (Overall SNR: {overall_snr_db:.2f} dB)'
    fig1.suptitle(title_text, fontsize=16)

    # Calculate y limits for each row
    row_ylims = []
    for row_idx in range(3):
        row_signals = [predicted_leads[LEAD_NAMES_BY_ROW[row_idx][col_idx]] for col_idx in range(4)]
        ymin = min(np.min(s) for s in row_signals)
        ymax = max(np.max(s) for s in row_signals)
        row_ylims.append((ymin, ymax))

    for row_idx in range(3):
        for col_idx in range(4):
            lead_name = LEAD_NAMES_BY_ROW[row_idx][col_idx]
            signal = predicted_leads[lead_name]
            t = np.arange(len(signal))

            ax = axes[row_idx, col_idx]

            # Set title with SNR if available
            lead_title = lead_name if lead_name != 'II_short' else 'II'
            if lead_snrs is not None and lead_name in lead_snrs and lead_name != 'II_short':
                lead_title = f'{lead_title} ({lead_snrs[lead_name]:.1f} dB)'

            # Map II_short to II for ground truth lookup
            gt_lead_name = 'II' if lead_name == 'II_short' else lead_name

            if ground_truth is not None and gt_lead_name in ground_truth.columns:
                # For small lead plots, only use as many ground truth samples as we have in prediction
                # (e.g., for II_short, use first 1/4 of the full Lead II ground truth)
                truth_signal = ground_truth[gt_lead_name].dropna().values[:len(signal)]
                t_truth = np.arange(len(truth_signal))
                ax.plot(t_truth, truth_signal, linewidth=0.8, color='red', alpha=0.7, label='Ground Truth')

                # Align prediction to ground truth
                aligned_signal = align_signals(truth_signal, signal, max_shift=len(truth_signal) * 0.2)
                ax.plot(t_truth, aligned_signal, linewidth=0.8, color='blue', label='Prediction (aligned)')
            else:
                # Plot prediction without alignment
                ax.plot(t, signal, linewidth=0.8, color='blue', label='Prediction')

            ax.set_title(lead_title, fontsize=14)
            ax.set_ylim(row_ylims[row_idx])
            ax.grid(True, alpha=0.3)
            if ground_truth is not None:
                ax.legend(fontsize=10)

    plt.tight_layout()
    if show:
        plt.show()

    # Full Lead II
    fig2, ax = plt.subplots(1, 1, figsize=(16, 3))
    full_lead_ii = predicted_leads['II']
    t = np.arange(len(full_lead_ii))

    title_text = f'Full Lead II'
    if lead_snrs is not None and 'II' in lead_snrs:
        title_text += f' ({lead_snrs["II"]:.1f} dB)'

    # Plot ground truth if available
    if ground_truth is not None and 'II' in ground_truth.columns:
        truth_signal = ground_truth['II'].dropna().values
        t_truth = np.arange(len(truth_signal))
        ax.plot(t_truth, truth_signal, linewidth=0.8, color='red', alpha=0.7, label='Ground Truth')

        # Align prediction to ground truth
        aligned_signal = align_signals(truth_signal, full_lead_ii, max_shift=len(truth_signal) * 0.2)
        ax.plot(t_truth, aligned_signal, linewidth=0.8, color='blue', label='Prediction (aligned)')
    else:
        # Plot prediction without alignment
        ax.plot(t, full_lead_ii, linewidth=0.8, color='blue', label='Prediction')
    ax.set_title(title_text, fontsize=14)
    ax.grid(True, alpha=0.3)
    if ground_truth is not None:
        ax.legend(fontsize=10)
    plt.tight_layout()
    if show:
        plt.show()


def plot_all_stages(image, normalised, rectified, predicted_leads, image_id, type_id,
                    ground_truth=None, overall_snr_db=None, lead_snrs=None,
                    output_dir='validation_output/plots', show=True):
    """
    Plot all three stages, save to files, and optionally display.

    Args:
        image: Original input image (RGB)
        normalised: Stage 0 normalized image (RGB)
        rectified: Stage 1 rectified image (RGB)
        predicted_leads: dict mapping lead_name -> signal array
        image_id: Image ID string
        type_id: Type ID string
        ground_truth: Optional ground truth DataFrame
        overall_snr_db: Optional overall SNR in dB
        lead_snrs: Optional dict mapping lead_name -> SNR in dB (pre-calculated)
        output_dir: Directory to save plots
        show: If True, display plots on screen
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    image_name = f'{image_id}-{type_id}'

    # Stage 0: Normalization
    plot_stage0_result(image, normalised, f'Stage 0: {image_name}', show=show)
    plt.savefig(f'{output_dir}/stage0-{image_id}-{type_id}.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # Stage 1: Rectification
    plot_stage1_result(normalised, rectified, f'Stage 1: {image_name}', show=show)
    plt.savefig(f'{output_dir}/stage1-{image_id}-{type_id}.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # Stage 2: Signal extraction
    plot_stage2_result(predicted_leads, f'Stage 2: {image_name}', ground_truth, overall_snr_db, lead_snrs, show=show)
    plt.savefig(f'{output_dir}/stage2-{image_id}-{type_id}.png', dpi=100, bbox_inches='tight')
    plt.close('all')


def calculate_lead_snr(truth_signal, pred_signal, fs):
    """
    Calculate SNR for a single lead.

    Args:
        truth_signal: np.ndarray - ground truth signal
        pred_signal: np.ndarray - predicted signal
        fs: Sampling frequency

    Returns:
        float: SNR in dB for this lead
    """
    # Align prediction to ground truth
    aligned_pred = align_signals(truth_signal, pred_signal, int(fs * MAX_TIME_SHIFT))

    # Compute power
    p_signal, p_noise = compute_power(truth_signal, aligned_pred)

    # Compute SNR
    snr = compute_snr(p_signal, p_noise)
    return 10 * np.log10(snr)


def build_submission_rows(predicted_leads, image_id, lead_records):
    """
    Build submission rows from predicted leads in competition format.

    Args:
        predicted_leads: dict mapping lead_name -> signal array
        image_id: Image ID
        lead_records: DataFrame with lead metadata (id, lead, fs, number_of_rows)

    Returns:
        list: Submission rows in format [{'id': ..., 'value': ...}, ...]
    """
    submission_rows = []

    for _, record in lead_records.iterrows():
        lead_name = record['lead']
        num_rows = int(record['number_of_rows'])

        # Get prediction
        if lead_name not in predicted_leads:
            continue

        pred_signal = predicted_leads[lead_name]
        pred_signal = interpolate_signal_to_length(pred_signal, num_rows)

        # Create rows in competition format
        for t in range(num_rows):
            row_id = f'{image_id}_{t}_{lead_name}'
            submission_rows.append({'id': row_id, 'value': float(pred_signal[t])})

    return submission_rows


def build_solution_rows(ground_truth, image_id, lead_records):
    """
    Build solution rows from ground truth in competition format.

    Args:
        ground_truth: pd.DataFrame with ground truth signals
        image_id: Image ID
        lead_records: DataFrame with lead metadata (id, lead, fs, number_of_rows)

    Returns:
        list: Solution rows in format [{'id': ..., 'fs': ..., 'value': ...}, ...]
    """
    solution_rows = []

    for _, record in lead_records.iterrows():
        lead_name = record['lead']
        num_rows = int(record['number_of_rows'])
        fs = record['fs']

        # Get ground truth
        if lead_name not in ground_truth.columns:
            continue

        truth_signal = ground_truth[lead_name].dropna().values[:num_rows]

        # Create rows in competition format
        for t in range(num_rows):
            row_id = f'{image_id}_{t}_{lead_name}'
            solution_rows.append({'id': row_id, 'fs': fs, 'value': float(truth_signal[t])})

    return solution_rows


def load_ground_truth(image_id, data_dir):
    """
    Load ground truth CSV for a training image.

    Args:
        image_id: Image ID
        data_dir: Path to PhysioNet data directory

    Returns:
        pd.DataFrame: Ground truth signals with columns for each lead
    """
    truth_path = f'{data_dir}/train/{image_id}/{image_id}.csv'
    return pd.read_csv(truth_path)


def save_validation_csvs(solution_df, submission_df, image_id, type_id, output_dir, predicted_leads=None, save_ii_short=False):
    """
    Save solution and submission CSVs for debugging/analysis.

    Args:
        solution_df: Solution DataFrame with ground truth
        submission_df: Submission DataFrame with predictions
        image_id: Image ID
        type_id: Type ID
        output_dir: Directory to save CSV files
        predicted_leads: Optional dict of lead signals (for saving II_short)
        save_ii_short: If True, save II_short lead to separate file
    """
    os.makedirs(output_dir, exist_ok=True)
    solution_df.to_csv(f'{output_dir}/sol-{image_id}-{type_id}.csv', index=False)
    submission_df.to_csv(f'{output_dir}/pred-{image_id}-{type_id}.csv', index=False)

    # Optionally save II_short lead separately (for Stage 3 training)
    if save_ii_short and predicted_leads and 'II_short' in predicted_leads:
        ii_short_path = f'{output_dir}/pred-{image_id}-{type_id}_II_short.csv'
        ii_short_df = pd.DataFrame({'II_short': predicted_leads['II_short']})
        ii_short_df.to_csv(ii_short_path, index=False)


def calculate_image_lead_snrs(predicted_leads, ground_truth, fs):
    """
    Calculate SNR for each lead in an image.

    Args:
        predicted_leads: dict mapping lead_name -> signal array
        ground_truth: pd.DataFrame with ground truth signals
        fs: Sampling frequency

    Returns:
        dict: Mapping of lead_name -> SNR in dB for this image
    """
    image_lead_snrs = {}
    for lead_name in LEADS:
        if lead_name not in ground_truth.columns or lead_name not in predicted_leads:
            continue

        pred_signal = predicted_leads[lead_name]
        truth_signal = ground_truth[lead_name].dropna().values[:len(pred_signal)]

        lead_snr_db = calculate_lead_snr(truth_signal, pred_signal, fs)
        image_lead_snrs[lead_name] = lead_snr_db

    return image_lead_snrs


def build_image_stats_row(image_id, type_id, overall_snr_db, image_lead_snrs):
    """
    Build statistics row for a single image.

    Args:
        image_id: Image ID
        type_id: Type ID
        overall_snr_db: Overall image SNR in dB
        image_lead_snrs: dict mapping lead_name -> SNR in dB for this image

    Returns:
        dict: Statistics row with image_id, type_id, overall, and per-lead SNRs
    """
    stats_row = {
        'image_id': image_id,
        'type_id': type_id,
        'overall': overall_snr_db
    }
    # Add per-lead SNRs
    for lead_name in LEADS:
        stats_row[lead_name] = image_lead_snrs.get(lead_name, None)
    return stats_row


def print_validation_summary(type_snrs, lead_snrs):
    """
    Print summary statistics for validation run.

    Args:
        type_snrs: dict mapping type_id -> list of SNR values in dB
        lead_snrs: dict mapping lead_name -> list of SNR values in dB
    """
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)

    # Overall average across all images
    all_image_snrs = []
    for type_id in type_snrs:
        all_image_snrs.extend(type_snrs[type_id])
    overall_avg_linear = np.mean([10 ** (db / 10) for db in all_image_snrs])
    overall_avg_db = 10 * np.log10(overall_avg_linear)
    print(f'\nOverall Average SNR: {overall_avg_db:.2f} dB (n={len(all_image_snrs)} images)')

    print('\nAverage SNR by Image Type:')
    for type_id in sorted(type_snrs.keys()):
        # Convert dB to linear, average, convert back to dB
        avg_linear = np.mean([10 ** (db / 10) for db in type_snrs[type_id]])
        avg_db = 10 * np.log10(avg_linear)
        print(f'  Type {type_id}: {avg_db:.2f} dB (n={len(type_snrs[type_id])})')

    print('\nAverage SNR by Lead:')
    for lead_name in LEADS:
        if lead_name in lead_snrs:
            # Convert dB to linear, average, convert back to dB
            avg_linear = np.mean([10 ** (db / 10) for db in lead_snrs[lead_name]])
            avg_db = 10 * np.log10(avg_linear)
            print(f'  {lead_name:>3}: {avg_db:.2f} dB (n={len(lead_snrs[lead_name])})')


def plot_snr_by_image_type(type_snrs, output_dir):
    """
    Generate and save box plot for SNR by image type.

    Args:
        type_snrs: dict mapping type_id -> list of SNR values in dB
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    type_data = [type_snrs[type_id] for type_id in sorted(type_snrs.keys())]
    type_labels = sorted(type_snrs.keys())
    ax.boxplot(type_data, labels=type_labels)
    ax.set_xlabel('Image Type')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR Distribution by Image Type')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary-snr-by-type.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_snr_by_lead(lead_snrs, output_dir):
    """
    Generate and save box plot for SNR by lead.

    Args:
        lead_snrs: dict mapping lead_name -> list of SNR values in dB
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    lead_data = [lead_snrs[lead] for lead in LEADS if lead in lead_snrs]
    lead_labels = [lead for lead in LEADS if lead in lead_snrs]
    ax.boxplot(lead_data, labels=lead_labels)
    ax.set_xlabel('Lead', fontsize=12)
    ax.set_ylabel('SNR (dB)', fontsize=12)
    ax.set_title('SNR Distribution by Lead', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary-snr-by-lead.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


def save_image_stats(image_stats, output_dir):
    """
    Save per-image statistics to CSV.

    Args:
        image_stats: List of statistics rows (dicts)
        output_dir: Directory to save CSV file
    """
    stats_df = pd.DataFrame(image_stats)
    stats_df.to_csv(f'{output_dir}/stats.csv', index=False)

# %%
def process_stage0(image, stage0_net, device='cuda', float_type=None):
    """
    Run Stage 0: Normalize image by detecting keypoints and applying homography.

    Args:
        image: np.ndarray of input ECG image (RGB)
        stage0_net: Loaded Stage 0 model
        device: 'cuda' or 'cpu'
        float_type: torch dtype for mixed precision

    Returns:
        tuple: (normalised, keypoint, homography)
            normalised: np.ndarray of normalized image (RGB)
            keypoint: detected keypoints
            homography: homography matrix

    Raises:
        Exception: If stage 0 processing fails
    """
    batch = image_to_batch(image)

    with torch.amp.autocast('cuda', dtype=float_type):
        with torch.no_grad():
            output = stage0_net(batch)
            rotated, keypoint = output_to_predict(image, batch, output)
            normalised, keypoint, homography = normalise_by_homography(rotated, keypoint)

    torch.cuda.empty_cache()

    return normalised, keypoint, homography

# %%
def process_stage1(normalised, stage1_net, device='cuda', float_type=None, num_tta=4):
    """
    Run Stage 1: Grid detection and rectification with TTA.

    Args:
        normalised: np.ndarray of normalized image from stage0 (RGB)
        stage1_net: Loaded Stage 1 model
        device: 'cuda' or 'cpu'
        float_type: torch dtype for mixed precision
        num_tta: Number of TTA variations (default 4)

    Returns:
        tuple: (rectified, gridpoint_xy)
            rectified: np.ndarray of rectified image (RGB)
            gridpoint_xy: detected grid points

    Raises:
        Exception: If stage 1 processing fails
    """
    # Prepare batch
    batch = {
        'image': torch.from_numpy(np.ascontiguousarray(normalised.transpose(2, 0, 1))).unsqueeze(0),
    }

    # TTA loop - brightness/contrast variations (compatible with training augmentations)
    marker_logit_sum = 0
    gridpoint_logit_sum = 0
    gridhline_logit_sum = 0
    gridvline_logit_sum = 0

    for trial in range(num_tta):
        crop = batch['image'].clone().float()

        # Apply brightness/contrast adjustments (preserves spatial structure)
        if trial == 0:
            pass  # Original
        elif trial == 1:
            crop = (crop * 1.1).clamp(0, 255)  # Brighter
        elif trial == 2:
            crop = (crop * 0.9).clamp(0, 255)  # Darker
        elif trial == 3:
            # Higher contrast (scale around mean)
            mean = crop.mean()
            crop = ((crop - mean) * 1.2 + mean).clamp(0, 255)

        crop = crop.byte()

        with torch.amp.autocast('cuda', dtype=float_type):
            with torch.no_grad():
                output = stage1_net({'image': crop})

                # Convert probabilities to logits (inverse softmax/sigmoid)
                eps = 1e-7
                marker_logit = torch.log(output['marker'].clamp(min=eps))
                gridpoint_logit = torch.log(output['gridpoint'].clamp(min=eps, max=1-eps) /
                                           (1 - output['gridpoint'].clamp(min=eps, max=1-eps)))
                gridhline_logit = torch.log(output['gridhline'].clamp(min=eps))
                gridvline_logit = torch.log(output['gridvline'].clamp(min=eps))

                # Accumulate logits
                marker_logit_sum = marker_logit_sum + marker_logit.float()
                gridpoint_logit_sum = gridpoint_logit_sum + gridpoint_logit.float()
                gridhline_logit_sum = gridhline_logit_sum + gridhline_logit.float()
                gridvline_logit_sum = gridvline_logit_sum + gridvline_logit.float()

    # Average logits and convert back to probabilities
    output = {
        'marker': torch.softmax(marker_logit_sum / num_tta, dim=1),
        'gridpoint': torch.sigmoid(gridpoint_logit_sum / num_tta),
        'gridhline': torch.softmax(gridhline_logit_sum / num_tta, dim=1),
        'gridvline': torch.softmax(gridvline_logit_sum / num_tta, dim=1),
    }

    # Get grid points and rectify image
    with torch.amp.autocast('cuda', dtype=float_type):
        with torch.no_grad():
            gridpoint_xy, more = stage1_output_to_predict(normalised, batch, output)
            rectified = rectify_image(normalised, gridpoint_xy)

    torch.cuda.empty_cache()

    return rectified, gridpoint_xy

# %%
def process_stage2(rectified, signal_length, stage2_net, device='cuda', float_type=None):
    """
    Run Stage 2: Extract ECG signal from rectified image.

    Args:
        rectified: np.ndarray of rectified image from stage1 (RGB)
        signal_length: Expected signal length (from metadata)
        stage2_net: Loaded Stage 2 model
        device: 'cuda' or 'cpu'
        float_type: torch dtype for mixed precision

    Returns:
        series: np.ndarray of shape (4, signal_length) containing extracted signals
            Row 0: I, aVR, V1, V4 (concatenated, split into 4 quarters)
            Row 1: II, aVL, V2, V5 (concatenated, split into 4 quarters)
            Row 2: III, aVF, V3, V6 (concatenated, split into 4 quarters)
            Row 3: II (full length)

    Raises:
        Exception: If stage 2 processing fails
    """
    # Rectified coordinate frame parameters
    x0, x1 = 0, 2176
    y0, y1 = 0, 1696
    zero_mv = [703.5, 987.5, 1271.5, 1531.5]
    mv_to_pixel = 79.0
    t0, t1 = 118, 2080

    # Crop rectified image
    crop = rectified[y0:y1, x0:x1]
    batch = {
        'image': torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).unsqueeze(0),
    }

    # Run stage2 model
    with torch.amp.autocast('cuda', dtype=float_type):
        with torch.no_grad():
            output = stage2_net(batch)
            pixel = output['pixel'].data.cpu().numpy()[0]

    # Convert pixel predictions to series
    series_in_pixel = pixel_to_series(pixel[..., t0:t1], zero_mv, signal_length)
    series = (np.array(zero_mv).reshape(4, 1) - series_in_pixel) / mv_to_pixel
    series = filter_series_by_limits(series)

    torch.cuda.empty_cache()

    return series

# %%
# Unified 3-stage pipeline for ECG image processing
# Consolidates processing logic used by both inference and validation

# Configuration
KAGGLE_DIR = '/kaggle/input/physionet-ecg-image-digitization'
WEIGHT_DIR = '/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLOAT_TYPE = torch.float32

# Load models once at module level
print('Loading models...')
stage0_net = Stage0Net(pretrained=False)
stage0_net = load_net(stage0_net, f'{WEIGHT_DIR}/stage0-last.checkpoint.pth')
stage0_net.to(DEVICE)

stage1_net = Stage1Net(pretrained=False)
stage1_net = load_net(stage1_net, f'{WEIGHT_DIR}/stage1-last.checkpoint.pth')
stage1_net.to(DEVICE)

stage2_net = Stage2Net(pretrained=False)
stage2_net = load_net(stage2_net, f'{WEIGHT_DIR}/stage2-00005810.checkpoint.pth')
stage2_net.to(DEVICE)
print('Models loaded.')


def process_image_pipeline(image, lead_records):
	"""
	Run full 3-stage pipeline on an ECG image and extract individual leads.

	Args:
		image: Input ECG image (RGB numpy array)
		lead_records: DataFrame with metadata for this image

	Returns:
		tuple: (predicted_leads, normalised, rectified)
			predicted_leads: dict mapping lead_name -> signal array for all 12 leads
			normalised: Stage 0 normalized image (for visualization)
			rectified: Stage 1 rectified image (for visualization)

	Raises:
		Exception: If any stage fails
	"""
	# Stage 0: Normalize image
	normalised, keypoint, homography = process_stage0(image, stage0_net, DEVICE, FLOAT_TYPE)

	# Stage 1: Rectify image using detected grid
	rectified, gridpoint_xy = process_stage1(normalised, stage1_net, DEVICE, FLOAT_TYPE)

	# Stage 2: Extract ECG signal from rectified image
	# Get expected signal length from metadata
	signal_length = lead_records[lead_records['lead'] == 'II'].iloc[0]['number_of_rows']
	series = process_stage2(rectified, signal_length, stage2_net, DEVICE, FLOAT_TYPE)

	# Extract individual leads from 4-row series format
	predicted_leads = extract_leads_from_series(series)

	return predicted_leads, normalised, rectified

# %%
# Inference on test images and submission creation

# Load test metadata
test_df = pd.read_csv(f'{KAGGLE_DIR}/test.csv')
test_df['id'] = test_df['id'].astype(str)

# Get unique image IDs from test set
image_ids = test_df['id'].unique().tolist()

# Build submission
submission_rows = []
total_images = len(image_ids)

for n, image_id in enumerate(image_ids):
	# Get lead records for this image
	lead_records = test_df[test_df['id'] == image_id].copy()

	filename = f'{KAGGLE_DIR}/test/{image_id}.png'
	print(f'\r Processing {n+1}/{total_images}: {image_id}', end='', flush=True)

	try:
		# Read image
		image = cv2.imread(filename, cv2.IMREAD_COLOR_RGB)

		# Run 3-stage pipeline
		predicted_leads, normalised, rectified = process_image_pipeline(image, lead_records)

		# Build submission rows for this image
		submission_rows.extend(build_submission_rows(predicted_leads, image_id, lead_records))

	except Exception as e:
		print(f'\nFailed on {image_id}: {e}')
		continue

print('\n\nBuilding submission DataFrame...')
submission_df = pd.DataFrame(submission_rows)

# Save submission
submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f'Submission saved to {submission_path}')
print(f'Total rows: {len(submission_df)}')

# %%
# Validation on training images with ground truth scoring

# Skip validation during competition rerun
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
	print('Skipping validation during competition rerun')
else:
	# Define which images to validate
	train_df = pd.read_csv(f'{KAGGLE_DIR}/train.csv')
	train_df['id'] = train_df['id'].astype(str)
	image_ids = train_df['id'].unique()[:10].tolist()  # 90 images

	# Initialize tracking for summary statistics
	type_snrs = {type_id: [] for type_id in TYPE_IDS}
	lead_snrs = {lead: [] for lead in LEADS}
	image_stats = []  # Per-image statistics for stats.csv

	# Process validation images
	for n, (filename, lead_records) in enumerate(validation_subset(image_ids, TYPE_IDS, KAGGLE_DIR)):
		print(f'\nProcessing {n+1}: {filename}', flush=True)

		# Extract image_id and type_id from filename
		image_name = os.path.basename(filename)
		type_id = image_name.split('-')[1].replace('.png', '')
		image_id = lead_records['id'].iloc[0]

		# Read image
		image = cv2.imread(filename, cv2.IMREAD_COLOR_RGB)

		# Run 3-stage pipeline
		try:
			predicted_leads, normalised, rectified = process_image_pipeline(image, lead_records)

			# Load ground truth
			ground_truth = load_ground_truth(image_id, KAGGLE_DIR)

			# Build submission and solution rows for this image
			submission_rows = build_submission_rows(predicted_leads, image_id, lead_records)
			solution_rows = build_solution_rows(ground_truth, image_id, lead_records)

			# Convert to dataframes
			submission_df = pd.DataFrame(submission_rows)
			solution_df = pd.DataFrame(solution_rows)

			# Calculate overall image SNR using competition scoring method
			fs = lead_records['fs'].iloc[0]
			overall_snr_db = score(solution_df, submission_df, 'id')
			print(f'Overall SNR: {overall_snr_db:.2f} dB')

			# Save solution and submission CSVs for this image
			save_validation_csvs(solution_df, submission_df, image_id, type_id, 'validation_output')

			# Calculate SNR for each lead
			image_lead_snrs = calculate_image_lead_snrs(predicted_leads, ground_truth, fs)
			for lead_name, lead_snr_db in image_lead_snrs.items():
				lead_snrs[lead_name].append(lead_snr_db)

			# Plot all stages (show first 10 images only)
			show_plots = (n < 10)
			plot_all_stages(image, normalised, rectified, predicted_leads, image_id, type_id,
			                ground_truth, overall_snr_db, image_lead_snrs, show=show_plots)

			# Track SNR by type (store dB values)
			type_snrs[type_id].append(overall_snr_db)

			# Collect stats for this image
			stats_row = build_image_stats_row(image_id, type_id, overall_snr_db, image_lead_snrs)
			image_stats.append(stats_row)

		except Exception as e:
			print(f'Failed on {filename}: {e}')
			continue

	# Print summary statistics
	print_validation_summary(type_snrs, lead_snrs)

	# Box plots
	plot_snr_by_image_type(type_snrs, 'validation_output/plots')
	plot_snr_by_lead(lead_snrs, 'validation_output/plots')

	# Save per-image statistics to CSV
	save_image_stats(image_stats, 'validation_output')
	print(f'Saved per-image statistics to validation_output/stats.csv ({len(image_stats)} images)')

	print('Validation complete!')


