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
    build_main_page_general_assistant()


def main():
    st.set_page_config(
        page_title="OTPP-GPT Dashboard",
        page_icon="✅",
        layout="wide",
    )
    st.html("../css/dashboard_styles.html")
    #st.markdown(page_bg_img, unsafe_allow_html=True)
    # ----------------------------------

    #--- Data for forecasting tab ----
    st.session_state.forecast_dataset_url = "../data/otpp/ev_charging/load_profile_ev_charging_by_location_days.csv"

    # read csv from a URL

    st.title("OTPP-GPT: Forecasting & Support")


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

   
            with forecast_tab:
                build_main_page_timeseries_forecasting()

            with doc_assist_tab:
                build_doc_assistant_tab()
        #time.sleep(4)
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

if __name__=='__main__':
    main()