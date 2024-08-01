import time  # to simulate a real time data, time loop
from pathlib import Path

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation

#--- streamlit and other UI imports 
import streamlit as st  # 🎈 data web app development
from streamlit_folium import st_folium
import streamlit_authenticator as stauth
from streamlit_timeline import timeline

import altair as alt
import folium
from folium.plugins import Realtime, MarkerCluster
from streamlit_folium import st_folium
from folium import JsCode

import pandas_geojson as pdg

import plotly.graph_objects as go

import yaml
from yaml.loader import SafeLoader

#--- llm imports 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain import LLMChain, PromptTemplate

#---- local imports ----
from utils.utils import *
from utils.dashboard_utils import *
from prompts.prompt_template import *
from scripts.ragapp import  check_sentence_hallucination, \
                            query_chain, get_session_chat_history, \
                            process_documents, load_documents, \
                            split_documents, load_vector_db, \
                            update_vector_db


REFRESH_TIMER = 2
# ---- SET UP THE LLMS ----
def build_llm_infrastructure():
    system_prompt = '''
    You are a bot who specializes on reading tabular data, summarizing them and providing insights.
    '''

    st.session_state.llm = 'llama3.1'
    setup_llms()
    setup_llm_chains()


# ---- FUNCTIONS FOR GRID OVERVIEW TAB ------
# ---- Draw realtime map ----- 

 
def draw_realtime_map():

    m = folium.Map(location=[st.session_state.cur_data_df['latitude'].mean(), st.session_state.cur_data_df['longitude'].mean()], zoom_start=10)
    source_anomaly= 'http://localhost:8000/anomalous.geojson'
    container = MarkerCluster(icon_create_function=icon_create_function).add_to(m)
    pt_layer_func = JsCode('''(f, latlng) => { 
                                var rad = f.properties.units*2
                                var popup_msg = 'Plant Name ' + f.properties.name + "<br>" + 'Plant Capacity :' + f.properties.units
                                return L.circleMarker(latlng, {radius: rad, fillOpacity: 0.4, color: '#cf1313', fillColor: '#cf1313', interactive: true}).bindPopup(popup_msg); }
                           
                           ''')
    realtime_layer_anomaly = Realtime(
        source_anomaly,  # Local URL to the GeoJSON file
        start=True,  # Automatically start refreshing
        get_feature_id=JsCode("(f) => { return f.properties.objectID}"),
        remove_missing=True,
        container=container,
        point_to_layer=pt_layer_func,
        interval=2000
    )
    realtime_layer_anomaly.add_to(m)
    st_folium(m, height=500, use_container_width=True)

# ----- Plot historic line chart for a given KPI -----

@st.experimental_fragment(run_every=REFRESH_TIMER)
def plot_historic_line_chart(historic_chart_kpi, df_historic_weekly_minmax):
    st.markdown("<h2 style='text-align: center; color: blue;'> Daily Trend </h2>", unsafe_allow_html=True)
    weekly_kpi_data = get_weekly_data(st.session_state.full_data_df, historic_chart_kpi, st.session_state.timestamps[-1])
    print(weekly_kpi_data)
    kpi_lines = alt.Chart(weekly_kpi_data).mark_line().encode(x='timestamp', y=alt.Y(f'{historic_chart_kpi}').title(historic_chart_kpi))
    pred_band = (alt.Chart(df_historic_weekly_minmax).mark_area(opacity=0.4, color= 'blue').encode(alt.X("timestamp").title("Hour"), 
                                                        y=alt.Y(f'{historic_chart_kpi}_max:Q').title(""),
                                                        y2=alt.Y2(f'{historic_chart_kpi}_min:Q').title("")))
    st.altair_chart((kpi_lines+pred_band), use_container_width=True)


# -----  Plot instantaneous barchart ---- 

@st.experimental_fragment(run_every=REFRESH_TIMER)
def plot_instantaneous_barchart(bar_chart_kpi):
    print("CUR TImestep: ", st.session_state.timestamps[-1])
    bar_chart_max = {'co2_emissions' : 20, 'reservoir_level' : 80, 'water_flow_rate' : 3300, 'total_energy_output' : 1000}
    st.markdown("<h2 style='text-align: center; color: blue;'> KPIs monitoring </h2>", unsafe_allow_html=True)
    realtime_bar = alt.Chart(st.session_state.cur_data_df).mark_bar().encode(
        x='name',
        y=alt.Y(bar_chart_kpi),
        color=alt.condition(
            f'datum.{bar_chart_kpi} > {bar_chart_max[bar_chart_kpi]}',
            alt.value('orange'),
            alt.value('steelblue')
        )
    )
    st.altair_chart(realtime_bar, use_container_width=True)


# ----- Writing the summarization columns -----

@st.experimental_fragment(run_every=REFRESH_TIMER)
def write_anomalies():
    t = st.session_state.timestamps[-1]
    st.markdown("<h2 style='text-align: center; color: blue;'> Anomalies registered </h2>", unsafe_allow_html=True)
    ano_col1, ano_col2, ano_col3, ano_col4 = st.columns(4)
    with ano_col1:
        st.html(f'<span class="anomaly_cards"></span>')
        power_ano = sum(st.session_state.cumm_data_df['anomaly_total_energy_output'])
        st.markdown("<h5 style='text-align: center; color: black;'>Power output</h5>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: center; color: red;'> {power_ano} </h6>", unsafe_allow_html=True)
    with ano_col2:
        st.html(f'<span class="anomaly_cards"></span>')
        reserv_ano = sum(st.session_state.cumm_data_df['anomaly_reservoir_level'])
        st.markdown("<h5 style='text-align: center; color: black;'>Reservoir level</h5>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: center; color: red;'> {reserv_ano} </h6>", unsafe_allow_html=True)
    with ano_col3:
        st.html(f'<span class="anomaly_cards"></span>')
        co2_ano = sum(st.session_state.cumm_data_df['anomaly_co2_emissions'])
        st.markdown("<h5 style='text-align: center; color: black;'>C02 level</h5>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: center; color: red;'> {co2_ano} </h6>", unsafe_allow_html=True)
    with ano_col4:
        st.html(f'<span class="anomaly_cards"></span>')
        flow_ano = sum(st.session_state.cumm_data_df['anomaly_water_flow_rate'])
        st.markdown("<h5 style='text-align: center; color: black;'>Water flow rate</h5>", unsafe_allow_html=True)
        st.write(f"<h6 style='text-align: center; color: red;'> {flow_ano} </h6>", unsafe_allow_html=True)

# --- Writing the llm summarization of current data ---- 
@st.experimental_fragment(run_every=REFRESH_TIMER)
def write_llm_summarization():
    if 'last_summarization_timestamp' not in st.session_state.keys():
        st.session_state.last_summarization_timestamp = None
    if st.session_state.last_summarization_timestamp!=st.session_state.timestamps[-1]:
        
        st.markdown("<h4 style='text-align: center; color: blue;'> System Summary </h4>", unsafe_allow_html=True)
        cur_df = st.session_state.cur_data_df[st.session_state.cur_data_df['anomaly']==1]
        max_val = min(5, len(cur_df))
        if max_val>0:
            with st.spinner('Generating summary . . .'):
                resp = st.session_state.llm_chain_summarizer.invoke({'table_data': cur_df.iloc[0:max_val].to_string()})  
                st.session_state.prev_timestamp_summary = parse_response(resp)
                st.write(parse_response(resp))
        else:
            st.write("All systems running smooth!")
        st.session_state.last_summarization_timestamp = st.session_state.timestamps[-1]
    else:
        st.markdown("<h4 style='text-align: center; color: blue;'> System Summary </h4>", unsafe_allow_html=True)
        st.write(st.session_state.prev_timestamp_summary)

# ----- INDIVIDUAL PLANT VIZ TAB -----
@st.experimental_fragment(run_every=REFRESH_TIMER)
def draw_plant_operational_status(plant_name):
    st.markdown("<h2 style='text-align: center; color: black;'> Operational Status </h2>", unsafe_allow_html=True)
    plant_status = st.session_state.full_data_df[st.session_state.full_data_df['name']==plant_name]['operational_status'].iloc[0]

    if plant_status==0:
        st.image('../assets/images/green_button.png',)
    if plant_status==1:
        st.image('../assets/images/red_button.png',)
    if plant_status==2:
        st.image('../assets/images/orange_button.png',)

@st.experimental_fragment(run_every=REFRESH_TIMER)
def plot_kpi_prediction_data(plant_name, pred_linechart_kpi):

    timesteps = 167
    t= st.session_state.timestamps[-1]
    plant_power_data_predict_mean = st.session_state.full_data_df[st.session_state.full_data_df['name']==plant_name][f'{pred_linechart_kpi}_predict_mean']
    plant_power_data_predict_std = st.session_state.full_data_df[st.session_state.full_data_df['name']==plant_name][f'{pred_linechart_kpi}_predict_std']
    kpi_data = list(st.session_state.full_data_df[st.session_state.full_data_df['name']==plant_name][f'{pred_linechart_kpi}'])[0:t+1]

    power_pred_df = pd.DataFrame()
    power_pred_df['timestep'] = np.arange(timesteps+1)

    #current data
    kpi_data.extend([float("NaN")]*(timesteps-t))
    power_pred_df[f'{pred_linechart_kpi}'] = kpi_data
    #pred mean
    pred_mean_nan = [float("NaN")]*t
    pred_mean_future = plant_power_data_predict_mean[t:]
    pred_mean_nan.extend(pred_mean_future)

    #pred_std
    pred_std_nan = [float("NaN")]*t
    pred_std_future = plant_power_data_predict_std[t:]
    pred_std_nan.extend(pred_std_future)

    #add cols to df
    power_pred_df[f'{pred_linechart_kpi}_pred_upper'] = np.array(pred_mean_nan)+np.array(pred_std_nan)
    power_pred_df[f'{pred_linechart_kpi}_pred_lower'] = np.array(pred_mean_nan)-np.array(pred_std_nan)
    power_pred_df[f'{pred_linechart_kpi}_pred_mean'] = pred_mean_nan

    st.markdown("<h2 style='text-align: center; color: black;'> Power generation forecasting </h2>", unsafe_allow_html=True)

    kpi_lines = alt.Chart(power_pred_df).mark_line().transform_fold(fold=[f'{pred_linechart_kpi}', f'{pred_linechart_kpi}_pred_mean']).mark_line().encode(x='timestep', y=alt.Y('value:Q').title("KPI"),color='key:N')

    pred_band = (alt.Chart(power_pred_df).mark_area(opacity=0.3, color= 'azure').encode(x='timestep', 
                                                        y=alt.Y(f'{pred_linechart_kpi}_pred_upper:Q').title(""),
                                                        y2=alt.Y2(f'{pred_linechart_kpi}_pred_lower:Q').title("")))
    st.altair_chart((kpi_lines+pred_band), use_container_width=True)

@st.experimental_fragment(run_every=REFRESH_TIMER)
def plot_gauge_chart(plant_name, 
                     kpi_name, 
                     plot_title= None, 
                     full_range=[0, 100], 
                     low_range=[0,25], 
                     mid_range=[26, 75], 
                     threshold=90):
        #gauge chart for reservoir level
        t = st.session_state.timestamps[-1]
        if len(st.session_state.timestamps)>1:
            t_prev = st.session_state.timestamps[-2]
        else:
            t_prev = st.session_state.timestamps[-1]

        res_level = st.session_state.cur_data_df[st.session_state.cur_data_df['name']==plant_name][kpi_name].iloc[0]
        past_res_level = st.session_state.cumm_data_df[(st.session_state.cumm_data_df['name']==plant_name) & \
                                                        (st.session_state.cumm_data_df['timestamp']==t_prev)][kpi_name].iloc[0]
   
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = res_level,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': plot_title},
            delta = {'reference': past_res_level},
            gauge = {'axis': {'range': full_range},
                    'steps' : [
                        {'range': low_range, 'color': "lightgray"},
                        {'range': mid_range, 'color': "gray"}],
                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}}))
        fig.update_layout(
            height=130,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=1, l=0, b=0, r=0, pad=0),
        )
        st.plotly_chart(fig, use_container_width=True)


@st.experimental_fragment(run_every=REFRESH_TIMER)
def build_indiv_plant_kpi_card(plant_name, kpi_name, 
                               plot_title= None, 
                                full_range=[0, 100], 
                                low_range=[0,25], 
                                mid_range=[26, 75], 
                                threshold=90):
    t = st.session_state.timestamps[-1]
    t_min = st.session_state.cumm_data_df['timestamp'].min()
    t_past_24 = max(t_min, t-24)

    kpi_past_24_data = st.session_state.cumm_data_df[(st.session_state.cumm_data_df['name']==plant_name) & \
                                                        (st.session_state.cumm_data_df['timestamp']<=t) & \
                                                          (st.session_state.cumm_data_df['timestamp']>=t_past_24)]
    
    kpi_max = kpi_past_24_data[kpi_name].max()
    kpi_min = kpi_past_24_data[kpi_name].min()

    #build the gauge chart
    plot_gauge_chart(plant_name, 
                    kpi_name, 
                    plot_title=None, 
                    full_range=full_range, 
                    low_range=low_range, 
                    mid_range=mid_range, 
                    threshold=threshold)
    st.write("Past 24 hr")

    daily_stat_min, daily_stat_max = st.columns(2)
    with daily_stat_min:
        st.write(f"Min\n{kpi_min}")
    with daily_stat_max:
        st.write(f"Max\n{kpi_max}")

    st.html(f'<span class="indiv_kpi_chart"></span>')
    fig_spark = go.Figure(go.Scatter(x=kpi_past_24_data['timestamp'], 
                                y=kpi_past_24_data[kpi_name],
                                mode='lines'))
    fig_spark.update_xaxes(visible=False, fixedrange=True)
    fig_spark.update_yaxes(visible=True, fixedrange=True)
    fig_spark.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=100,
        margin=dict(t=10, l=0, b=0, r=0, pad=0),
    )
    st.plotly_chart(fig_spark)
    
@st.experimental_fragment(run_every=REFRESH_TIMER)
def build_anomaly_timeline(plant_name):
    plant_data = st.session_state.cumm_data_df[st.session_state.cumm_data_df['name']==plant_name]
    plant_data.rename(columns={'datetime':'start_date'})
    print(plant_data)
    timeline(plant_data.to_json())

@st.experimental_fragment(run_every=REFRESH_TIMER)
def build_indiv_plant_tab():

    pred_linechart_kpi = 'total_energy_output'
    plant_name = st.selectbox('Select plant' , st.session_state.cur_data_df['name'].unique())

    indiv_plant_row1 = st.empty()
    indiv_plant_row2 = st.empty()
    with indiv_plant_row1.container(height=500, border=True):
        indiv_plant_row1_col1, indiv_plant_row1_col2 = st.columns([0.2, 0.8])

        with indiv_plant_row1_col1:
            
            draw_plant_operational_status(plant_name)

        with indiv_plant_row1_col2:

            plot_kpi_prediction_data(plant_name, pred_linechart_kpi)
            
    with indiv_plant_row2.container(height=500, border=True):
        

        indiv_plant_row2_col1, indiv_plant_row2_col2, indiv_plant_row2_col3 = st.columns(3, gap='large')
        
        with indiv_plant_row2_col1:
            
            #gauge chart for reservoir_level
            plot_gauge_chart(plant_name, 'reservoir_level', plot_title="Reservoir Level")
            
        with indiv_plant_row2_col2:
            
            #gauge chart for co2 emissions
            plot_gauge_chart(plant_name, 
                            'co2_emissions', 
                            plot_title='CO2 Emissions', 
                            full_range=[0, 24],
                            low_range=[0, 10],
                            mid_range=[10, 20],
                            threshold=22)

        with indiv_plant_row2_col3:
            
            #gauge chart for water_flow_level
            system_cap =  st.session_state.cur_data_df[st.session_state.cur_data_df['name']==plant_name]['capacity (mw)'].iloc[0]
            if system_cap < 10:
                water_flow_multiplier = 1

            elif system_cap < 100:
                water_flow_multiplier = 10

            else:
                water_flow_multiplier = 100

            gauge_range = [0, 70*water_flow_multiplier]
            light_gray_range = [0, 20*water_flow_multiplier]
            gray_range = [20*water_flow_multiplier, 50*water_flow_multiplier]
            threshold = 65*water_flow_multiplier

            plot_gauge_chart(plant_name, 
                            'water_flow_rate', 
                            plot_title='Water flow rate', 
                            full_range=gauge_range,
                            low_range=light_gray_range,
                            mid_range=gray_range,
                            threshold=threshold)


# ----- CHATBOT ASSISTANT TAB -----

def setup_llms():

    st.session_state.llm_model_chat = Ollama(model='llama3.1', system='You are a helpful question answering bot.')
    st.session_state.llm_model_instruct = Ollama(model='llama3.1', system="You are an LLM that is excellent at following instructions.")
    st.session_state.llm_dashboard_assistant = Ollama(model='llama3.1', system="You are a bot who specializes on reading tabular data, summarizing them and providing insights.")

def setup_llm_chains():

    #build the conversation chain
    conv_prompt = PromptTemplate(input_variables=['input', 'history'], template=CONV_PROMPT_TEMPLATE)
    st.session_state.conv_chain = LLMChain(llm=st.session_state.llm_model_chat, prompt=conv_prompt, output_key='answer')
    
    #build the rephrase chain 
    st.session_state.rephrase_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=RETRIEVE_REPHRASE_PROMPT)

    #build the document chain
    st.session_state.document_chain=create_stuff_documents_chain(st.session_state.llm_model_chat, DOCUMENT_CHAIN_PROMPT)

    #build the router chain
    router_prompt = PromptTemplate(
        input_variables=["input"], template=ROUTER_PROMPT_TEMPLATE_3
    )
    st.session_state.router_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=router_prompt, output_key='answer')

    #build the email writing chain
    email_prompt = PromptTemplate(input_variables=['input'], template=EMAIL_PROMPT_TEMPLATE)
    st.session_state.email_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=email_prompt, output_key='answer')    

    #build the table summarizer chain
    db_summarizer_prompt = PromptTemplate(
        input_variables=['input'], template=TABLE_SUMMARIZER_TEMPLATE
    )
    st.session_state.table_summarizer_chain = LLMChain(llm=st.session_state.llm_dashboard_assistant,
                                                       prompt=db_summarizer_prompt)
    
    #build the nlp to pandas retriever chain
    nl_to_pandas_prompt = PromptTemplate(
        input_variables=['input'], template=NL_TO_PANDAS_QUERY_TEMPLATE
    )
    st.session_state.pandas_query_chain = LLMChain(llm=st.session_state.llm_dashboard_assistant,
                                                       prompt=nl_to_pandas_prompt)
    
    #build the table data analyzer chain
    tabular_data_summarizer_prompt = PromptTemplate(
        input_variables=['table_data'], template=TABLE_SUMMARIZER_TEMPLATE
    )
    
    st.session_state.tabular_data_summarizer_chain = LLMChain(
                                llm=st.session_state.llm_dashboard_assistant,
                                prompt=tabular_data_summarizer_prompt
                                )


def query_chain():
    #run the email chain

    query_text = st.session_state.current_input
    update_vector_db()

    k = st.session_state.search_k if st.session_state.search_k else 3  
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": k})

    #use chains

    #check if retrieval is required
    resp = st.session_state.router_chain.invoke({'input': query_text})
    print("***RESPONSE QA : ", resp['answer'])
    is_qa = get_key_val_from_llm_json_string(resp['answer'], 'response')
    
    input_dict = {'input': query_text, 'chat_history': get_session_chat_history()}
    
    if is_qa.strip().lower()=='qa':

        #rephrase question using history
        resp = st.session_state.rephrase_chain.invoke(input_dict)
        resp_string = get_key_val_from_llm_json_string(resp['text'], 'rephrased_input')
        print("**********Rephrased input :", resp_string)
        #use response to retrieve relevant documents 
        docs = retriever.get_relevant_documents(resp_string)

        #get answer using relevant documents and question
        result = st.session_state.document_chain.invoke({'input':resp_string, 
                                                'context':docs})
        

        #annotate the response with hallucination information
        resp_sent, scores = check_sentence_hallucination(resp_string, docs, result, sample_size=5)
        anno_result = ""
        for sent, score in zip(resp_sent, scores):
            if score > 0.3: #0 is no hallu, 1 is hallu
                sent = f":red-background[{sent}]"
            anno_result += sent 
        #result = anno_result
        print("Scores ***************", scores)
        print("RESULT ***************", result)
        print("RESULT ***************", anno_result)

        st.session_state.response = anno_result
        st.session_state.response_context = docs

    if is_qa.strip().lower()=='conv':
        result = st.session_state.conv_chain.invoke(input_dict)
        anno_result = result['answer']
        st.session_state.response = result
        st.session_state.response_context = ""    
    
    if is_qa.strip().lower()=='writing':
        resp = st.session_state.rephrase_chain.invoke(input_dict)
        resp_string = get_key_val_from_llm_json_string(resp['text'], 'rephrased_input')
        print("**********Rephrased input :", resp_string)
        result = st.session_state.email_chain.invoke({'input':resp_string})
        anno_result = result['answer']
        st.session_state.response = result
        st.session_state.response_context = ""  

    if is_qa.strip().lower()=='power':

        #do stuff
        resp = st.session_state.pandas_query_chain.invoke(input_dict)
        db_query = get_key_val_from_llm_json_string(resp['text'], 'query')
        print("PANDA Query : ", db_query)
        tab_data = eval(db_query)
        result = st.session_state.tabular_data_summarizer_chain({'table_data': tab_data})
        print("REsponse from TABLE :", result)
        anno_result = parse_response(result)
        #st.session_state.response = parse_response(result)
        st.session_state.response_context = ""  
 
    
    #save the query in the chat history
    st.session_state.messages.append({"speaker" : "user", "content": query_text})
    
    # rel_sources = [doc.metadata['source'] for doc in docs]
    # rel_pages = [doc.metadata['page'] for doc in docs]
    # rel_data_resp = f'\n Relevant information can be found in the following documents : {" ".join(rel_sources)}'
    st.session_state.messages.append({"speaker" : "AI",
                                    "content": anno_result})


def build_chat_window():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.chat_input(placeholder = 'Enter query here ...', 
                on_submit=query_chain,
                key='current_input')
    chat_row = st.empty()
    #context_row = st.empty()
    with chat_row.container(height=450, border=True):
        #display the chat history so far
        for msg in st.session_state.messages:
            st.chat_message(msg['speaker']).markdown(msg['content'])

        #display the documents in the context used to come up with the answer
    # with context_row.container(height=200, border=True):
    #     if 'response_context' in st.session_state.keys():
    #         for doc in st.session_state.response_context:
    #             st.write(doc)


    
def build_doc_assistant_tab():

    with st.container(height=500):
        upload_doc_col, search_db_col = st.columns(2)
        with upload_doc_col:
            if 'messages' not in st.session_state.keys():
                st.session_state.messages = []
            
            st.session_state.source_docs = st.file_uploader(label="Upload Documents", 
                                                            type="pdf", 
                                                            accept_multiple_files=True)
            st.button("Submit documents", on_click=process_documents)
            k_list = [3,4,5,6,7]
            st.session_state.search_k = st.selectbox('No. of documents in context:', k_list)
            st.session_state.func = st.selectbox('Select Database', ['Public', 'Private'], index=0)
            st.session_state.vector_db = load_vector_db(st.session_state.vector_db_paths[st.session_state.func])

            with st.popover("Show files"):
                metadatas = st.session_state.vector_db.get()['metadatas']
                all_files = list(set([entry['source'] for entry in metadatas]))
                mark_down_text = ''
                for f in all_files:
                    mark_down_text+= '- '+f+'\n'
                print(mark_down_text)
                st.markdown(mark_down_text)

            uploaded_file = st.session_state.source_docs
        with search_db_col:
            st.write('Set filters')
            with st.form("Set filters"):
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.db_filter_p_name = st.selectbox('Plant name', st.session_state.cumm_data_df['name'].unique())
                    st.session_state.db_filter_s_ts = st.selectbox('Start timestamp', st.session_state.cumm_data_df['timestamp'].unique())
                    st.session_state.db_filter_e_ts = st.selectbox('End timestamp', st.session_state.cumm_data_df['timestamp'].unique())
                with col2:
                    st.session_state.db_filter_inc_energy = st.checkbox('Energy anomaly', value=False)
                    st.session_state.db_filter_inc_flowrate = st.checkbox('Flowrate anomaly', value=False)
                    st.session_state.db_filter_inc_reservoir = st.checkbox('Reservoir level anomaly', value=False)

                retrieve_data = st.form_submit_button("Fetch data")
                if retrieve_data:
                    #build the query
                    query = f'(st.session_state.cumm_data_df["name"]=="{st.session_state.db_filter_p_name}") & \
                                (st.session_state.cumm_data_df["timestamp"]>={st.session_state.db_filter_s_ts}) & \
                                    (st.session_state.cumm_data_df["timestamp"]<={st.session_state.db_filter_e_ts})'
                    if st.session_state.db_filter_inc_energy:
                        query+= ' & (st.session_state.cumm_data_df["anomaly_total_energy_output"]==1)'
                    if st.session_state.db_filter_inc_flowrate:
                        query+= ' & (st.session_state.cumm_data_df["anomaly_water_flow_rate"]==1)'
                    if st.session_state.db_filter_inc_reservoir:
                        query+= ' & (st.session_state.cumm_data_df["anomaly_reservoir_level"]==1)'
                    
                    final_query = f'st.session_state.cumm_data_df[{query}]'
                    print("FINAL QUERY : ", final_query)
                    st.dataframe(eval(final_query))


@st.experimental_fragment(run_every=REFRESH_TIMER)
def get_data() -> pd.DataFrame:
    st.session_state.cur_data_df = pd.read_csv(st.session_state.cur_dataset_url, index_col=[0])

    t = st.session_state.cur_data_df['timestamp'].iloc[0]

    if 'timestamps' not in st.session_state.keys():
        st.session_state.timestamps = []
    
    if t not in st.session_state.timestamps:
        st.session_state.timestamps.append(t)
        if 'cumm_data_df' not in st.session_state.keys():
            st.session_state.cumm_data_df = st.session_state.cur_data_df
        else:
            st.session_state.cumm_data_df = pd.concat([st.session_state.cumm_data_df, st.session_state.cur_data_df], ignore_index=True)

    print("UNIQUE TIMESTAMPS :", st.session_state.cumm_data_df['timestamp'].unique())

def get_data_full():
    st.session_state.full_data_df = pd.read_csv(st.session_state.dataset_url)

#---SET BACKGROUND --- 


def main():
    st.set_page_config(
        page_title="EnergyGPT Dashboard",
        page_icon="✅",
        layout="wide",
    )
    st.html("../styles.html")
    st.html("<link rel='stylesheet' type='text/css' href='../leaflet.css' />")
    st.markdown(page_bg_img, unsafe_allow_html=True)
    # ----------------------------------


    #--- EXTERNAL DB INFORMATION  ----
    st.session_state.dataset_url = "../data/dashboard/dashboard_monitoring_data.csv"
    st.session_state.cur_dataset_url = "../data/dashboard/dashboard_monitoring_data_per_hr.csv"
    st.session_state.kpi_list = ['total_energy_output', 'reservoir_level', 'water_flow_rate', 'co2_emissions']

    #--- TODO : Change the way the VECTOR_DB_PATHS  work in dashboard.py and ragapp.py ---

    st.session_state.vector_db_paths = {
                    'Public' : Path('../vectorstores/energy_public'), 
                    'Private' : Path('../vectorstores/energy_private'),
                        }
    # read csv from a URL

    get_data()
    get_data_full()
    # ----------------------------

    build_llm_infrastructure()

    st.title("EnergyGPT: Monitoring and assistance")


    if 'count' not in st.session_state:
        st.session_state.count = 0
    #  ------------USER AUTHENTICAION-----------------

    with open('../assets/authentication/credentials.yaml', 'r', encoding='utf-8') as file:
        cred_data = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        cred_data['credentials']['names'],
        cred_data['credentials']['usernames'],
        cred_data['credentials']['passwords'],
        cred_data['cookie']['name'],
        cred_data['cookie']['key'],
        cred_data['cookie']['expiry_days'],
    )

    name, authentication_status, username = authenticator.login('Login', 'sidebar')
    if authentication_status:
        st.session_state.count+=1
        if st.session_state.count==1:
            pass 
        else:
        
            authenticator.logout('Logout', 'main')
            st.write(f"Welcome :blue[{name}]")
            
            grid_overview_tab, indiv_plant_tab, doc_assist_tab = st.tabs(['Grid overview', 'Plant overview', 'Assistant'])

            #---- SET UP THE PAGE STRUCTURE ---

            with grid_overview_tab:
                #create the map container
                st.markdown("<h2 style='text-align: center; color: black;'> Anomaly detection </h2>", unsafe_allow_html=True)

                with st.container(height=635):
                    map_col, chat_col = st.columns([0.7, 0.3])

                    with map_col:
                        st.markdown("<h2 style='text-align: center; color: black;'> Interactive map </h2>", unsafe_allow_html=True)

                        draw_realtime_map()
                    with chat_col:
                        st.markdown("<h2 style='text-align: center; color: black;'> Assistant </h2>", unsafe_allow_html=True)
                        build_chat_window()


                col1, col2 = st.columns(2)

                with col1:
                    historic_chart_kpi = st.selectbox("Select KPI", st.session_state.kpi_list, key='line_chart')
                with col2:
                    bar_chart_kpi = st.selectbox("Select KPI", st.session_state.kpi_list, key='bar_chart')

                #calculate daily averages of different KPIS
                df_grouped = st.session_state.full_data_df.groupby(['timestamp']).agg({historic_chart_kpi:['mean']})
                kpi_week_data = np.array(df_grouped[historic_chart_kpi]['mean']).reshape((-1 ,24))
                kpi_week_min = kpi_week_data.min(axis=0)
                kpi_week_max = kpi_week_data.max(axis=0)

                df_historic_weekly_minmax = pd.DataFrame()
                df_historic_weekly_minmax[f'{historic_chart_kpi}_min'] = kpi_week_min 
                df_historic_weekly_minmax[f'{historic_chart_kpi}_max'] = kpi_week_max 
                df_historic_weekly_minmax['timestamp'] = np.arange(24)

                grid_overview_row1 = st.empty()
                grid_overview_row2 = st.empty()

                # creating a single-element container
                with grid_overview_row1.container(height=500):

                    # create two columns for charts
                    fig_col1, fig_col2 = st.columns(2)
                    with fig_col1:
                        #line chart over dayc
                        plot_historic_line_chart(historic_chart_kpi, df_historic_weekly_minmax)

                    with fig_col2:
                        #barchart with instantaneous readings
                        plot_instantaneous_barchart(bar_chart_kpi)


                with grid_overview_row2.container(height=250, border=True):
                    
                    anomaly_col, summary_col = st.columns(2, gap='small')

                    with anomaly_col:
                        write_anomalies()
                        
                    with summary_col.container(height = 220, border=True):
                        #write_llm_summarization()
                        pass

            with indiv_plant_tab:

                pred_linechart_kpi = 'total_energy_output'
                plant_name = st.selectbox('Select plant' , st.session_state.cur_data_df['name'].unique())

                indiv_plant_row1 = st.empty()
                indiv_plant_row2 = st.empty()
                with indiv_plant_row1.container(height=500, border=True):
                    indiv_plant_row1_col1, indiv_plant_row1_col2 = st.columns([0.2, 0.8])

                    with indiv_plant_row1_col1:
                        
                        draw_plant_operational_status(plant_name)

                    with indiv_plant_row1_col2:

                        plot_kpi_prediction_data(plant_name, pred_linechart_kpi)
                        
                with indiv_plant_row2.container(height=500, border=True):
                    

                    indiv_plant_row2_col1, indiv_plant_row2_col2, indiv_plant_row2_col3 = st.columns(3, gap='large')
                    
                    with indiv_plant_row2_col1.container(border=True):
                        
                        #gauge chart for reservoir_level
                        #plot_gauge_chart(plant_name, 'reservoir_level', plot_title="Reservoir Level")
                        st.markdown("<h4 style='text-align: center; color: black;'> Reservoir Level </h4>", unsafe_allow_html=True)
                        build_indiv_plant_kpi_card(plant_name, 'reservoir_level',plot_title="Reservoir Level")

                    with indiv_plant_row2_col2.container(border=True):
                        
                        #gauge chart for co2 emissions
                        st.markdown("<h4 style='text-align: center; color: black;'> CO2 Emissions </h4>", unsafe_allow_html=True)

                        build_indiv_plant_kpi_card(plant_name, 
                                        'co2_emissions', 
                                        plot_title='CO2 Emissions', 
                                        full_range=[0, 24],
                                        low_range=[0, 10],
                                        mid_range=[10, 20],
                                        threshold=22)

                    with indiv_plant_row2_col3.container(border=True):
                        
                        st.markdown("<h4 style='text-align: center; color: black;'> Water flow rate </h4>", unsafe_allow_html=True)

                        #gauge chart for water_flow_level
                        system_cap =  st.session_state.cur_data_df[st.session_state.cur_data_df['name']==plant_name]['capacity (mw)'].iloc[0]
                        if system_cap < 10:
                            water_flow_multiplier = 1

                        elif system_cap < 100:
                            water_flow_multiplier = 10

                        else:
                            water_flow_multiplier = 100

                        gauge_range = [0, 70*water_flow_multiplier]
                        light_gray_range = [0, 20*water_flow_multiplier]
                        gray_range = [20*water_flow_multiplier, 50*water_flow_multiplier]
                        threshold = 65*water_flow_multiplier

                        build_indiv_plant_kpi_card(plant_name, 
                                        'water_flow_rate', 
                                        plot_title='Water flow rate', 
                                        full_range=gauge_range,
                                        low_range=light_gray_range,
                                        mid_range=gray_range,
                                        threshold=threshold)

                #build_anomaly_timeline(plant_name)

            with doc_assist_tab:
                
                build_doc_assistant_tab()
        time.sleep(4)
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

if __name__=='__main__':
    main()