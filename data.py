import numpy as np

def clean_data(data):
    return np.nan_to_num(data)

def normalize_data(data):
    return data / 100