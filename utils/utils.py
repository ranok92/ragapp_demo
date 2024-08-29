import json
import ipdb 
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