import time  # to simulate a real time data, time loop
from pathlib import Path
import os 

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation

#--- streamlit and other UI imports 
import streamlit as st  # 🎈 data web app development
from streamlit_folium import st_folium
import streamlit_authenticator as stauth
from streamlit_timeline import st_timeline
from streamlit_extras.stylable_container import stylable_container
import altair as alt
import folium
from folium.plugins import Realtime, MarkerCluster
from streamlit_folium import st_folium
from folium import JsCode

import pandas_geojson as pdg

import plotly.graph_objects as go

import yaml
from yaml.loader import SafeLoader
import statistics
import datetime
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
from timeseries_forecasting import *
from st_aggrid import AgGrid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from anomaly_detection import *

with open('../assets/openai_api_key.txt', 'r') as f:
    key = f.read()
os.environ["OPENAI_API_KEY"]=key

REFRESH_TIMER = 2
# ---- FUNCTIONS FOR GRID OVERVIEW TAB ------
# ----- CHATBOT ASSISTANT TAB -----
def get_session_anomaly_chat_history():
    chat_list = st.session_state.messages_anomaly 
    chat_history = []
    for conv in chat_list:
        if conv['speaker']=="user":
            chat_history.append(HumanMessage(content=conv['content']))
        if conv['speaker']=='AI':
            chat_history.append(AIMessage(content=conv['content']))
    return chat_history

def setup_llms():

    st.session_state.llm_model_chat = Ollama(model='llama3.1', system='You are a helpful question answering bot.')
    st.session_state.llm_model_instruct = Ollama(model='llama3.1', format='json', system="You are an LLM who is logical and is excellent at following instructions.")
    # st.session_state.llm_dashboard_assistant = Ollama(model='llama3.1', format='json', system="You are a bot who specializes on reading tabular data, summarizing them and providing insights.")

    #OPENAI
    st.session_state.llm_dashboard_assistant = ChatOpenAI(
                                                model="gpt-4o-mini" ,
                                                temperature=0,
                                                max_retries=2,
                                                # api_key="...",
                                                # base_url="...",
                                                # organization="...",
                                                # other params...
                                            )

def setup_llm_chains():

    #build the conversation chain
    conv_prompt = PromptTemplate(input_variables=['input', 'history'], template=CONV_PROMPT_TEMPLATE)
    st.session_state.conv_chain = LLMChain(llm=st.session_state.llm_model_chat, 
                                           prompt=conv_prompt, 
                                           output_key='answer')
    
    #build the rephrase chain 
    rephrase_prompt = PromptTemplate(input_variables=['input', 'chat_history'], template=RETRIEVE_REPHRASE_PROMPT)
    st.session_state.rephrase_chain = LLMChain(llm=st.session_state.llm_dashboard_assistant, 
                                               prompt=rephrase_prompt)

    #build the document chain
    st.session_state.document_chain=create_stuff_documents_chain(st.session_state.llm_model_chat,
                                                                  DOCUMENT_CHAIN_PROMPT)

    #build the router chain
    router_prompt = PromptTemplate(
        input_variables=["input"], template=ROUTER_PROMPT_TEMPLATE_BASIC
    )
    st.session_state.router_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=router_prompt, output_key='answer')

    #build the email writing chain
    email_prompt = PromptTemplate(input_variables=['input'], template=EMAIL_PROMPT_TEMPLATE)
    st.session_state.email_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=email_prompt, output_key='answer')    

    #build the nlp to pandas retriever chain
    nl_to_pandas_prompt = PromptTemplate(
        input_variables=['input'], template=NL_TO_PANDAS_QUERY_TEMPLATE_OUTAGE
    )
    st.session_state.pandas_query_chain = LLMChain(llm=st.session_state.llm_dashboard_assistant,
                                                       prompt=nl_to_pandas_prompt)
    
    #build the table data analyzer chain
    tabular_data_summarizer_prompt = PromptTemplate(
        input_variables=['input', 'info_from_db'], template=TABLE_SUMMARIZER_TEMPLATE_OUTAGE
    )
    
    st.session_state.tabular_data_summarizer_chain = LLMChain(
                                llm=st.session_state.llm_dashboard_assistant,
                                prompt=tabular_data_summarizer_prompt
                                )

def query_chain_anomaly_assistant():
    #run the email chain

    query_text = st.session_state.current_input_anomaly
    chat_history = get_session_anomaly_chat_history()
    #use chains
    #rephrase using historic context
    resp = st.session_state.rephrase_chain.invoke({'input': query_text, 'chat_history': chat_history})
    print("**********Rephrased input whole:", resp)

    resp_string = get_key_val_from_llm_json_string(resp['text'], 'rephrased_input')
    print("**********Rephrased input :", resp_string)


    #check if retrieval is required
    router_samples = 3
    router_resp_list = []
    for i in range(router_samples):

        router_resp = st.session_state.router_chain.invoke({'input': resp_string})
        print("***RESPONSE QA : ", router_resp['answer'])
        router_resp_list.append(get_key_val_from_llm_json_string(router_resp['answer'], 'response'))
    
    is_qa = statistics.median(router_resp_list)
    print("***  ROUTER RESPONSE : ", router_resp_list)
    
    if is_qa.strip().lower()=='conv':
        result = st.session_state.conv_chain.invoke({'input': resp_string, 
                                                     'chat_history': get_session_anomaly_chat_history()})
        anno_result = result['answer']
        st.session_state.response = result
        st.session_state.response_context = ""    
    
    if is_qa.strip().lower()=='writing':
        result = st.session_state.email_chain.invoke({'input':resp_string})
        anno_result = result['answer']
        
        st.session_state.response = anno_result
        st.session_state.response_context = ""  

    if is_qa.strip().lower()=='outage':

        #do stuff
        resp = st.session_state.pandas_query_chain.invoke({'input': resp_string})
        print("RESP: ", resp)
        db_query = get_key_val_from_llm_json_string(resp['text'], 'query')
        print("PANDA Query : ", db_query)
        table_data = eval(db_query)
        print("The retrieved TABLE :", table_data)
        result = st.session_state.tabular_data_summarizer_chain.invoke({'input': resp_string, 
                                                                        'info_from_db': table_data})
        print("REsponse from TABLE :", result)
        # if result['text'].strip()[0]!='{':
        #     result['text'] = '{'+result['text']+'}'
        # result_json_data = json.loads(result['text'])
        # anno_result = result_json_data['summary']+result_json_data['thoughts']
        anno_result = result['text']
        #st.session_state.response = parse_response(result)
        st.session_state.response_context = ""  
 
    
    #save the query in the chat history
    st.session_state.messages_anomaly.append({"speaker" : "user", "content": query_text})
    
    # rel_sources = [doc.metadata['source'] for doc in docs]
    # rel_pages = [doc.metadata['page'] for doc in docs]
    # rel_data_resp = f'\n Relevant information can be found in the following documents : {" ".join(rel_sources)}'
    st.session_state.messages_anomaly.append({"speaker" : "AI",
                                    "content": anno_result})


def build_chat_window_anomaly():
    if "messages_anomaly" not in st.session_state:
        st.session_state.messages_anomaly = []
    st.chat_input(placeholder = 'Enter query here ...', 
                on_submit=query_chain_anomaly_assistant,
                key='current_input_anomaly')
    chat_row = st.empty()
    #context_row = st.empty()
    with chat_row.container(height=450, border=True):
        #display the chat history so far
        for msg in st.session_state.messages_anomaly:
            st.chat_message(msg['speaker']).markdown(msg['content'])

        #display the documents in the context used to come up with the answer
    # with context_row.container(height=200, border=True):
    #     if 'response_context' in st.session_state.keys():
    #         for doc in st.session_state.response_context:
    #             st.write(doc)



def build_chat_window_assistant():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.chat_input(placeholder = 'Enter query here ...', 
                on_submit=query_chain,
                key='current_input')
    chat_row_assistant = st.empty()
    #context_row = st.empty()
    with chat_row_assistant.container(height=450, border=True):
        #display the chat history so far
        for msg in st.session_state.messages:
            st.chat_message(msg['speaker']).markdown(msg['content'])

        #display the documents in the context used to come up with the answer
    # with context_row.container(height=200, border=True):
    #     if 'response_context' in st.session_state.keys():
    #         for doc in st.session_state.response_context:
    #             st.write(doc)



def build_data_filter_window():
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
            AgGrid(eval(final_query))
    
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
            build_chat_window_assistant()


#---- Build solar forecast tab

def build_forecast_tab():
    print("running forecast tab")
    setup_llms_forecast()
    setup_llm_chains_forecast()
    # read csv from a github repo
    st.session_state.forecast_dataset_url = "../data/dashboard/solar_powerplant_forecasting_data.csv"
    st.session_state.full_forecast_data_df = get_data_forecast()
    plant_names = st.session_state.full_forecast_data_df['name'].unique()
    pred_linechart_kpi = 'total_energy_output'
    pred_df = None
    #design the UI
    with stylable_container(
        key='forecast_header',
        css_styles='''
        {
            text-align: center;
            padding: 20px;
            background: #4b6cb7;
            color: white;
            border-radius: 10px;
        }
''',
    ):
        st.markdown(f'<h1 style="color: white;"> Forecast Dashboard </h1>', unsafe_allow_html=True)
    col1, col2, = st.columns([0.27, 0.73])
    with col1:
        param_form_container = st.container(height=800, border=True)
        run_eval_container = st.container(height=250, border=True)

    with col2:
        pred_stats_container = st.container(height=280, border=True)
        
        pred_plot_container = st.container(height=770, border=True)


        with pred_plot_container:
            st.markdown("<h3 style='text-align: center; color: black;'> Forecast Plot </h3>", unsafe_allow_html=True)
    
    #with pred_stats_container:
        

    with col1:
        with param_form_container:
            build_param_selection_form()
            with st.popover(":headphones:", help='Model Consultant'):
                build_chat_window_forecast_assistant()

        with run_eval_container:
            with st.form("Evaluate on ", border=False):
                st.markdown(f'<h3 style="color:black;text-align:center">Evaluate on: </h2>', unsafe_allow_html=True)
                plant_name = st.selectbox('Select Plant', plant_names)
                predict_button = st.form_submit_button("Run Predition")
            if predict_button:
                with pred_plot_container:
                    pred_df = plot_kpi_prediction_data(plant_name, pred_linechart_kpi)

    with col2:
            with pred_stats_container:
                st.markdown("<h3 style='text-align: center; color: black;'> Forecast Error </h3>", unsafe_allow_html=True)

                if pred_df is not None:
                    show_error_metrics(pred_df, pred_linechart_kpi)

def main():
    st.set_page_config(
        page_title="EnergyGPT Dashboard",
        page_icon="✅",
        layout="wide",
    )
    st.html("../css/dashboard_styles.html")
    #st.markdown(page_bg_img, unsafe_allow_html=True)
    # ----------------------------------


    #--- EXTERNAL DB INFORMATION  ----
    st.session_state.dataset_url = "../data/dashboard/outage_monitoring_data.csv"
    st.session_state.cur_dataset_url = "../data/dashboard/outage_monitoring_data_per_hr.csv"
    st.session_state.kpi_list = ['total_energy_output', 'reservoir_level', 'water_flow_rate', 'co2_emissions']

    #--- TODO : Change the way the VECTOR_DB_PATHS  work in dashboard.py and ragapp.py ---

    #vector DBs for work efficiency improvement
    st.session_state.vector_db_paths = {
                    'Public' : Path('../vectorstores/energy_public'), 
                    'Private' : Path('../vectorstores/energy_private'),
                        }
    
    #loading db for anomaly solution assistant 

    anomaly_soln_vector_db_path = Path('../vectorstores/powerplant_anomaly_solutions')
    st.session_state.anomaly_soln_vector_db = load_vector_db(anomaly_soln_vector_db_path)
    
    # read csv from a URL

    get_data_anomaly()
    get_data_full_anomaly()
    # ----------------------------
    setup_llms_anomaly()
    setup_llm_chains_anomaly()

    st.title("EnergyGPT: Monitoring & Assistance")


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
            with st.container():
                st.html('<span class="logout"></span>')
                authenticator.logout('Logout', 'main')
            st.write(f"Welcome :blue[{name}]")
            
            grid_overview_tab, forecast_tab, doc_assist_tab = st.tabs([':bar_chart: Anomaly Detection', ':factory: Energy Forecasting', ':headphones: Assistant'])

            #---- SET UP THE PAGE STRUCTURE ---

            with grid_overview_tab:
                #create the header container
                header_container =  stylable_container(key='anomaly_header',
                                        css_styles=''' 
                                        {
                                            text-align: center;
                                            padding: 20px;
                                            background: #4b6cb7;
                                            color: white;
                                            border-radius: 10px;
                                        }
                                        ''')
                map_and_chat_container = st.container(height=700, border=False)
                grid_overview_container = st.container(height=550, border=False)
                
                with header_container:
                    st.markdown("<h2 style='font-family: sans-serif; text-align: center; color: white;'> Anomaly Detection Dashboard</h2>", unsafe_allow_html=True)

                with map_and_chat_container:
                    map_col, chat_col = st.columns([0.7, 0.3])

                    with map_col:
                        write_latest_update_time()
                        draw_realtime_map()
                    with chat_col:
                        st.markdown("<h2 style='text-align: center; color: #453030;'> Assistant </h2>", unsafe_allow_html=True)
                        build_chat_window_anomaly()


                with grid_overview_container:

                    # create two columns for charts
                    fig_col1, fig_col2 = st.columns([0.7,0.3])
                    with fig_col1:
                        #line chart over dayc
                        #plot_historic_line_chart(historic_chart_kpi, df_historic_weekly_minmax)
                        plot_outage_occurance_linechart()
                    with fig_col2:
                        #barchart with instantaneous readings
                        write_outages()

                # with grid_overview_row2.container(height=250, border=True):
                    
                #     anomaly_col, summary_col = st.columns(2, gap='small')

                #     with anomaly_col:
                #         write_anomalies()
                        
                #     with summary_col.container(height = 220, border=True):
                #         #write_llm_summarization()
                #         pass

            with forecast_tab:
                build_forecast_tab()

            with doc_assist_tab:
                build_doc_assistant_tab()
        #time.sleep(4)
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

if __name__=='__main__':
    main()