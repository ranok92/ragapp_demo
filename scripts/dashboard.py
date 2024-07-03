import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import re
import matplotlib.pyplot as plt

import altair as alt
import folium
from streamlit_folium import st_folium
from folium.plugins import Realtime
from streamlit_folium import st_folium
from folium import JsCode
import pandas_geojson as pdg

def get_geojson_from_df(regular_df):
    '''
    returns a df that can be converted to geojson df
    regular df has the following columns 	
    location latitude longitude	timestep	
    kpi1 kpi1_anomaly kpi1_predict_mean	kpi1_predict_std	
    kpi2 kpi2_anomaly kpi2_predict_mean	kpi2_predict_std
    '''
    coord = [[long, lat] for long, lat in zip(regular_df['longitude'], regular_df['latitude'])]
    regular_df = regular_df.drop(columns=['latitude', 'longitude', 'timestep'])
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
    print(regular_df, geo_json)
    return geo_json


st.set_page_config(
    page_title="Telecom Dashboard",
    page_icon="✅",
    layout="wide",
)

# read csv from a github repo
dataset_url = "../data/dashboard_data.csv"
KPI_LIST = ['kpi1', 'kpi2']


# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_data()

# dashboard title
st.title("Telecommunications data tracking dashboard")

# top-level filters
location_filter = st.selectbox("Select a location", pd.unique(df["location"]))
kpi_filter = st.selectbox("Select a KPI", KPI_LIST)


#create the map container

# st.markdown("### Anomaly detection")
# #kpi_anomaly = alt.Chart(df_chart).mark_line().transform_fold(fold=[f'{kpi_filter}', f'{kpi_filter}_anomaly']).mark_line().encode(x='timestep', y=alt.Y('value:Q').title("KPI"),color='key:N')
# kpi_value = alt.Chart(df_chart).mark_line().encode(x='timestep', y=alt.Y(f'{kpi_filter}', title="KPI"))
# kpi_anomaly = alt.Chart(df_chart).mark_point(color='red').encode(x='timestep', y=f'{kpi_filter}_anomaly')
# st.altair_chart((kpi_value+kpi_anomaly), use_container_width=True)
with st.container(height=600):
    m = folium.Map(location=[ df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)
    source= 'http://localhost:8000/dashboard.geojson'

    realtime_layer = Realtime(
        source,  # Local URL to the GeoJSON file
        start=True,  # Automatically start refreshing
        get_feature_id=JsCode("(f) => { return f.properties.objectID; }"),
        point_to_layer=JsCode("(f, latlng) => { return L.circleMarker(latlng, {radius: 15, fillOpacity: 0.2}); }"),
        interval=500
    )
    realtime_layer.add_to(m)
    st_folium(m, height=600, use_container_width=True)

# creating a single-element container
graph_placeholder = st.empty()
df_placeholder = st.empty()
# dataframe filter
df_part = df[df["location"] == location_filter]

# near real-time / live feed simulation
timesteps = max(df['timestep'])
#print(df[df['timestep']==t][kpi_filter])




for t in range(timesteps):



    ###

    df_chart = pd.DataFrame()
    df_chart.insert(0, 'location', df_part['location'])

    #setup the line chart data 

    #data that already happened
    kpi_data_mean = list(df_part[f'{kpi_filter}'][0:t])
    kpi_data_mean.extend([float("NaN")]*(timesteps+1-t))
    df_chart.insert(1, f'{kpi_filter}', kpi_data_mean)

    #data that is supposed to happen
    #mean
    pred_mean_nan = [float("NaN")]*t
    kpi_data_pred_mean = list(df_part[f'{kpi_filter}_predict_mean'][t:])
    pred_mean_nan.extend(kpi_data_pred_mean)
    #std 
    pred_std_nan = [float("NaN")]*t
    kpi_data_pred_std = list(df_part[f'{kpi_filter}_predict_std'][t:])
    pred_std_nan.extend(kpi_data_pred_std)

    df_chart.insert(2, f'{kpi_filter}_pred_upper', list(np.array(pred_mean_nan)+np.array(pred_std_nan)))
    df_chart.insert(3, f'{kpi_filter}_pred_lower', list(np.array(pred_mean_nan)-np.array(pred_std_nan)))

    df_chart.insert(4, f'{kpi_filter}_pred_mean', pred_mean_nan)
    df_chart.insert(5,'timestep', df_part['timestep'])
    
    #anomaly
    kpi_data_ano = [float("NaN")]*(timesteps+1)
    for i in range(t):
        if df_part[f'{kpi_filter}_anomaly'].iloc[i]==1:
            kpi_data_ano[i] = df_part[f'{kpi_filter}'].iloc[i]

    print(df[(df[f'{kpi_filter}_anomaly']==1) & (df['timestep']==t)])
    anomaly_geo_data = get_geojson_from_df(df[(df[f'{kpi_filter}_anomaly']==1) & (df['timestep']==t)])
    pdg.save_geojson(anomaly_geo_data,'../data/geo_data/dashboard.geojson',indent=4)
    df_chart.insert(6, f'{kpi_filter}_anomaly', kpi_data_ano)


    # df["age_new"] = df["age"] * np.random.choice(range(1, 5))
    # df["balance_new"] = df["balance"] * np.random.choice(range(1, 5))

    # # creating KPIs
    # avg_age = np.mean(df["age_new"])

    # count_married = int(
    #     df[(df["marital"] == "married")]["marital"].count()
    #     + np.random.choice(range(1, 30))
    # )

    # balance = np.mean(df["balance_new"])

    with graph_placeholder.container(height=450):

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Time series forecasting")
            #st.line_chart(df_chart, y=[f'{kpi_filter}_pred_mean', f'{kpi_filter}'])
            # kpi_line = (alt.Chart(df_chart).mark_line(opacity=1, color='green').encode(x='timestep', y=alt.Y(f'{kpi_filter}', title="Actual")))
            # kpi_line_pred = (alt.Chart(df_chart).mark_line(opacity=0.5, color='pink').encode(x='timestep', y=f'{kpi_filter}_pred_mean'))
            kpi_lines = alt.Chart(df_chart).mark_line().transform_fold(fold=[f'{kpi_filter}', f'{kpi_filter}_pred_mean']).mark_line().encode(x='timestep', y=alt.Y('value:Q').title("KPI"),color='key:N')

            pred_band = (alt.Chart(df_chart).mark_area(opacity=0.3, color= 'azure').encode(x='timestep', 
                                                                y=alt.Y(f'{kpi_filter}_pred_upper:Q').title(""),
                                                                y2=alt.Y2(f'{kpi_filter}_pred_lower:Q').title("")))

            st.altair_chart((kpi_lines+pred_band), use_container_width=True)
    

        with fig_col2:
            st.markdown("### Anomaly detection")
            #kpi_anomaly = alt.Chart(df_chart).mark_line().transform_fold(fold=[f'{kpi_filter}', f'{kpi_filter}_anomaly']).mark_line().encode(x='timestep', y=alt.Y('value:Q').title("KPI"),color='key:N')
            kpi_value = alt.Chart(df_chart).mark_line().encode(x='timestep', y=alt.Y(f'{kpi_filter}', title="KPI"))
            kpi_anomaly = alt.Chart(df_chart).mark_point(color='red').encode(x='timestep', y=f'{kpi_filter}_anomaly')
            st.altair_chart((kpi_value+kpi_anomaly), use_container_width=True)
        
    with df_placeholder.container():
        st.markdown("### Detailed Data View")

        st.dataframe(df_part[df_part['timestep']<=t][['location','timestep',f'{kpi_filter}', f'{kpi_filter}_anomaly']])
    
    time.sleep(3)