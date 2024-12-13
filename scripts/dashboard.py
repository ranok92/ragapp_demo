import time  # to simulate a real time data, time loop
from pathlib import Path
import os 

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation

#--- streamlit and other UI imports 
import streamlit as st  # 🎈 data web app development
import streamlit_authenticator as stauth
from streamlit_extras.stylable_container import stylable_container
import yaml
from yaml.loader import SafeLoader

#---- local imports ----
from utils.utils import *
from utils.dashboard_utils import *
from prompts.prompt_template import *
from timeseries_forecasting import *
from anomaly_detection import *
from general_assistant import *

with open('../assets/openai_api_key.txt', 'r') as f:
    key = f.read()
os.environ["OPENAI_API_KEY"]=key

REFRESH_TIMER = 2
# ---- FUNCTIONS FOR GRID OVERVIEW TAB ------
# ----- CHATBOT ASSISTANT TAB -----

def build_chat_window_assistant():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.chat_input(placeholder = 'Enter query here ...', 
                on_submit=query_chain_general_assistant,
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
    

def build_doc_assistant_tab():
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
        st.markdown(f'<h2 style="color: white;"> Assistant </h2>', unsafe_allow_html=True)
    setup_llms_assistant()
    setup_llm_chains_assistant()
    load_vectordbs()    
    load_hallucination_detector()

    if "messages_gen_assist" not in st.session_state:
        st.session_state.messages_gen_assist = []

    col1, col2 = st.columns([0.3,0.7], gap="small")
    with col1:
        respose_gen_console =  st.container(height=350, border=True)
        document_management_console =  st.container(height=550, border=True)


    with respose_gen_console:
        build_chatbot_params_console()

    with col2:
        if st.session_state.show_supporting_docs:
            chat_window =  st.container(height=500,  border=False)  
            context_display_console = st.container(height=400,  border=True)  
        else:
            chat_window =  st.container(height=900,  border=True)  


    with chat_window:
        st.chat_input(placeholder = 'Ask me anything: From writing emails to finding answers from documents. ', 
                        on_submit=query_chain_general_assistant,
                        key='current_input')

        #display the chat history so far
        if st.session_state.show_supporting_docs:
            with st.container(height=400):
                for msg in st.session_state.messages_gen_assist:
                    st.chat_message(msg['speaker']).markdown(msg['content'])
        else:
            with st.container(height=800):
                for msg in st.session_state.messages_gen_assist:
                    st.chat_message(msg['speaker']).markdown(msg['content'])

    #display the documents in the context used to come up with the answer

    if st.session_state.show_supporting_docs:
        with context_display_console:
                build_context_display_window()

    with document_management_console:
        build_doc_management_console()

#---- Build solar forecast tab

def build_forecast_tab():
    print("running forecast tab")
    setup_llms_forecast()
    setup_llm_chains_forecast()
    st.session_state.full_forecast_data_df = get_data_forecast()
    plant_names = st.session_state.full_forecast_data_df['street_name'].unique()
    day_of_week = st.session_state.full_forecast_data_df['day'].unique()

    pred_linechart_kpi = 'load (kw)'
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
        st.markdown(f'<h2 style="color: white;"> Forecast Dashboard </h2>', unsafe_allow_html=True)
    col1, col2, = st.columns([0.27, 0.73])
    with col1:
        param_form_container = st.container(height=800, border=True)
        run_eval_container = st.container(height=340, border=True)

    with col2:
        pred_stats_container = st.container(height=340, border=True)
        
        pred_plot_container = st.container(height=800, border=True)


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
                day = st.selectbox('Select Day', day_of_week)

                predict_button = st.form_submit_button("Run Predition")
            if predict_button:
                with pred_plot_container:
                    pred_df = plot_kpi_prediction_data(plant_name, day, pred_linechart_kpi)

    with col2:
            with pred_stats_container:
                st.markdown("<h3 style='text-align: center; color: black;'> Forecast Error </h3>", unsafe_allow_html=True)

                if pred_df is not None:
                    show_error_metrics(pred_df, pred_linechart_kpi)

def main():
    st.set_page_config(
        page_title="DecisionGPT Dashboard",
        page_icon="✅",
        layout="wide",
    )
    st.html("../css/dashboard_styles.html")
    #st.markdown(page_bg_img, unsafe_allow_html=True)
    # ----------------------------------

    #--- Data for forecasting tab ----
    st.session_state.forecast_dataset_url = "../data/otpp/ev_charging/load_profile_ev_charging_by_location_days.csv"

    # read csv from a URL

    # get_data_anomaly()
    # get_data_full_anomaly()
    # # ----------------------------
    # setup_llms_anomaly()
    # setup_llm_chains_anomaly()

    st.title("FinanceGPT: Forecasting & Support")


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
            
            forecast_tab, doc_assist_tab = st.tabs([':chart_with_upwards_trend: Forecasting', ':headphones: Assistant'])

            #---- SET UP THE PAGE STRUCTURE ---

            # with grid_overview_tab:
            #     #create the header container
            #     header_container =  stylable_container(key='anomaly_header',
            #                             css_styles=''' 
            #                             {
            #                                 text-align: center;
            #                                 background: #4b6cb7;
            #                                 color: white;
            #                                 border-radius: 10px;
            #                             }
            #                             ''')
            #     map_and_chat_container = st.container(height=700, border=False)
            #     grid_overview_container = st.container(height=550, border=False)
                
            #     with header_container:
            #         st.markdown("<h2 style= 'text-align: center; color: white;'> Anomaly Detection Dashboard</h2>", unsafe_allow_html=True)

            #     with map_and_chat_container:
            #         map_col, chat_col = st.columns([0.7, 0.3])

            #         with map_col:
            #             write_latest_update_time()
            #             draw_realtime_map()
            #         with chat_col:
            #             st.markdown("<h2 style='text-align: center; color: #453030;'> Assistant </h2>", unsafe_allow_html=True)
            #             build_chat_window_anomaly()


            #     with grid_overview_container:

            #         # create two columns for charts
            #         fig_col1, fig_col2 = st.columns([0.7,0.3])
            #         with fig_col1:
            #             #line chart over dayc
            #             #plot_historic_line_chart(historic_chart_kpi, df_historic_weekly_minmax)
            #             plot_outage_occurance_linechart()
            #         with fig_col2:
            #             #barchart with instantaneous readings
            #             write_outages()

            #     # with grid_overview_row2.container(height=250, border=True):
                    
            #     #     anomaly_col, summary_col = st.columns(2, gap='small')

            #     #     with anomaly_col:
            #     #         write_anomalies()
                        
            #     #     with summary_col.container(height = 220, border=True):
            #     #         #write_llm_summarization()
            #     #         pass

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