import pandas_geojson as pdg
import pandas as pd
import numpy as np


def get_weekly_data(full_data, kpi_str, t):
    week = t//7
    df_week = full_data[(full_data['timestamp']>=week*7) & (full_data['timestamp']<=t)]
    df_ts = df_week.groupby(['timestamp']).agg({kpi_str:['mean']})
    df_alt_line = pd.DataFrame()
    df_alt_line[kpi_str]  = df_ts[kpi_str]['mean']
    df_alt_line['timestamp'] = np.arange(0, t-week*7+1)
    return df_alt_line

def parse_response(response):
    resp_text = response['text']
    summary_thoughts = resp_text.split("'summary'")[1].strip(" ':,.}{ ")
    summary = summary_thoughts.split("'thoughts'")[0].strip(" ':,. }{ ")

    thoughts = summary_thoughts.split("'thoughts'")[1].strip(" ':,. }{ ")
    resp_string = f'{summary}. {thoughts}'
    return resp_string

def get_geojson_from_df(regular_df, t):
    '''
    returns a df that can be converted to geojson df
    regular df has the following columns 	
    location latitude longitude	timestep	
    kpi1 kpi1_anomaly kpi1_predict_mean	kpi1_predict_std	
    kpi2 kpi2_anomaly kpi2_predict_mean	kpi2_predict_std
    '''
    coord = [[long, lat] for long, lat in zip(regular_df['longitude'], regular_df['latitude'])]
    regular_df = regular_df.drop(columns=['latitude', 'longitude', 'timestamp'])
    regular_df = regular_df.dropna(axis=1)
    #drop NA columns
    regular_df['type'] = ['Point']*len(regular_df)
    regular_df['objectID'] = [str(i) for i in range(len(regular_df))]

    property_cols = regular_df.columns
    regular_df['type'] = ['Point']*len(regular_df)
    regular_df['coordinates'] = coord
    geo_json = pdg.GeoJSON.from_dataframe(regular_df
                                     ,geometry_type_col='type'
                                     ,coordinate_col='coordinates'
                                     ,property_col_list=property_cols
                                     )
 
    print(len(regular_df), t)
    return geo_json


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-color: rgb(233,125,139);
background-image: linear-gradient(167deg, rgba(233,125,139,1) 0%, rgba(255,255,255,1) 20%)
}
</style>
"""

icon_create_function = '''
            function(cluster) {
            return L.divIcon({html: '<b style="text-align:center; font-size:20px; color: red">' + cluster.getChildCount() + '</b>',
                              className: 'marker-cluster marker-cluster-large',
                              iconSize: "auto"});
            }
            '''