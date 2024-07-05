import json
import ipdb 
import pandas_geojson as pdg

def get_key_val_from_llm_json_string(dict_text, key):
    dict_text_p1 = dict_text.split('{')[1].split('}')[0]
    dict_text_wh = '{'+dict_text_p1+'}'
    print("DICT :::::: ", dict_text_wh)
    return json.loads(dict_text_wh)[key]


