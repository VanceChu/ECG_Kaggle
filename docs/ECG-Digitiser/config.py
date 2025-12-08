IMAGE_TYPE = "png" # Image type for the dataset
FREQUENCY = 500 # Sampling frequency of the signals
DATASET_NAME = "Dataset500_Signals" # Name of the dataset
LONG_SIGNAL_LENGTH_SEC = 10 # Length in seconds of the full signal
SHORT_SIGNAL_LENGTH_SEC = 2.5 # Length in seconds of the cropped signal
SIGNAL_UNITS = "mV" # Units of the signal for the y-axis
FMT = '16' # Format of the signal
ADC_GAIN = 1000.0 # ADC gain of the signal
BASELINE = 0 # Baseline of the signal

# Mapping of the lead labels to num for the segmentation model
LEAD_LABEL_MAPPING = {
    "I": 1,
    "II": 2,
    "III": 3,
    "aVR": 4,
    "aVL": 5,
    "aVF": 6,
    "V1": 7,
    "V2": 8,
    "V3": 9,
    "V4": 10,
    "V5": 11,
    "V6": 12,
}

# If the absolute y value matters, use the following values
# to adjusts the vertical positioning of the cropped signal within the rotated image's height.
# A value closer to 1 shifts the signal toward the top, while a value closer to 0 shifts it toward the bottom.
Y_SHIFT_RATIO = {
    "I": 12.6 / 21.59,
    "II": 9 / 21.59,
    "III": 5.4 / 21.59,
    "aVR": 12.6 / 21.59,
    "aVL": 9 / 21.59,
    "aVF": 5.4 / 21.59,
    "V1": 12.59 / 21.59,
    "V2": 9 / 21.59,
    "V3": 5.4 / 21.59,
    "V4": 12.59 / 21.59,
    "V5": 9 / 21.59,
    "V6": 5.4 / 21.59,
    "full": 2.1 / 21.59,
}
