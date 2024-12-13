import json
import ipdb 
import numpy as np
import pandas_geojson as pdg

def get_key_val_from_llm_json_string(dict_text, key):
    dict_json = str_to_dict(dict_text)
    return dict_json[key]


def str_to_dict(str_data):
    str_data = str_data.strip("`json")
    if str_data.strip()[0]!='{':
        str_data = '{'+ str_data
    if str_data.strip()[-1]!='}':
        str_data = str_data +'}'
    return json.loads(str_data)


def symmetric_mape(actual, predicted):
    """
    Symmetric Mean Absolute Percentage Error
    Handles small values more robustly
    
    Args:
        actual (array-like): True values
        predicted (array-like): Predicted values
    
    Returns:
        float: sMAPE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero
    denominator = np.abs(actual) + np.abs(predicted)
    
    # Prevent division by zero by adding a small epsilon
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    
    smape = np.mean(2 * np.abs(actual - predicted) / denominator * 100)
    print(actual, predicted, (2 * np.abs(actual - predicted) / denominator * 100))
    
    return smape