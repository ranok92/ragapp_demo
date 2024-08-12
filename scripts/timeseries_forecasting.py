
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import streamlit as st  # 🎈 data web app development
from utils.utils import *
from utils.dashboard_utils import *
import json
import altair as alt 

from prompts.prompt_template import *
from langchain_community.llms import Ollama
from langchain import LLMChain, PromptTemplate

def get_data():
    return pd.read_csv(st.session_state.dataset_url)


def setup_llms():
    st.session_state.llm_dashboard_assistant = Ollama(model='llama3.1', 
                            system="You are a bot who is an expert on timeseries model prediction.")

def setup_llm_chains():

    #build the conversation chain
    pred_assistant_prompt = PromptTemplate(input_variables=['input', 'history'], template=PRED_ASSISTANT_PROMPT_TEMPLATE)
    st.session_state.assistant_chain = LLMChain(llm=st.session_state.llm_dashboard_assistant, prompt=pred_assistant_prompt, output_key='answer')
    
def query_chain():
    input_query = st.session_state.current_input
    st.session_state.messages.append({"speaker" : "user", "content": input_query})
    resp = st.session_state.assistant_chain.invoke({'input':input_query})
    # rel_sources = [doc.metadata['source'] for doc in docs]
    # rel_pages = [doc.metadata['page'] for doc in docs]
    # rel_data_resp = f'\n Relevant information can be found in the following documents : {" ".join(rel_sources)}'
    st.session_state.messages.append({"speaker" : "AI",
                                    "content": resp['answer']})



def build_chat_window_assistant():
    st.markdown(f'<h3 style="font-family:serif; color:black; text-align:center">Forecasting Assistant</h3>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.chat_input(placeholder = 'Enter query here ...', 
                on_submit=query_chain,
                key='current_input')
    chat_row_assistant = st.empty()
    #context_row = st.empty()
    with chat_row_assistant.container(height=230, border=True):
        #display the chat history so far
        for msg in st.session_state.messages:
            st.chat_message(msg['speaker']).markdown(msg['content'])


def build_data_filter_window():
    st.markdown(f'<h3 style="color:black;font-family:serif;text-align:center">Prediction Param Selection</h2>', unsafe_allow_html=True)
    param_select_form = st.form('Select params')
    with param_select_form:
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            features_selected = st.multiselect("Features to include", 
                                            ['Feature1', 'Feature2', 'Feature3', 'Feature4'],
                                            'Feature1')
            model_selected = st.selectbox("Pick a model", 
                                        ['Model 1', 'Model 2', 'Model 3'])
                
            add_normalization = st.toggle('Add normalization')
            add_dropout = st.toggle('Add dropout')

        with form_col2:
            forecasting_horizon = st.selectbox("Pick a prediction horizon", 
                                                ['1 hr', '1 day', '1 week', '1 month'])
            
            training_epochs = st.text_input('Training Epochs', 1000)
            learning_rate = st.text_input("Learning rate", 0.001)
            select_optimizer = st.selectbox("Pick an optimizer", 
                                            ['Opt1', 'Opt2', 'Opt3'])
        
        retrieve_data = st.form_submit_button("Set Params")

def plot_kpi_prediction_data(plant_name, pred_linechart_kpi):

    timesteps = 167
    t= 0
    plant_power_data_predict_mean = st.session_state.full_data_df[st.session_state.full_data_df['name']==plant_name][f'{pred_linechart_kpi}_predict_mean']
    plant_power_data_predict_std = st.session_state.full_data_df[st.session_state.full_data_df['name']==plant_name][f'{pred_linechart_kpi}_predict_std']
    kpi_data = list(st.session_state.full_data_df[st.session_state.full_data_df['name']==plant_name][f'{pred_linechart_kpi}'])[0:t+1]
    #kpi_data = []

    power_pred_df = pd.DataFrame()
    power_pred_df['hours'] = np.arange(timesteps+1)

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
    power_pred_df[f'{pred_linechart_kpi}_pred_mean'] = pred_mean_nan+np.random.rand(168)*30
    #power_pred_df[f'']

    multiplier = power_pred_df[f'{pred_linechart_kpi}_pred_mean'].mean()/5
    power_pred_df[f'{pred_linechart_kpi}_pred_upper'] = power_pred_df[f'{pred_linechart_kpi}_pred_mean']+np.array(pred_std_nan)+np.random.rand(168)*multiplier+multiplier/5
    power_pred_df[f'{pred_linechart_kpi}_pred_lower'] = power_pred_df[f'{pred_linechart_kpi}_pred_mean']-np.array(pred_std_nan)-np.random.rand(168)*multiplier-multiplier/5
    power_pred_df['mean_label'] = (timesteps+1)*['mean']
    power_pred_df['stddev_label'] = (timesteps+1)*['std deviation']

    st.markdown("<h2 style='text-align: center; color: black;'> Forecasted Power Generation </h2>", unsafe_allow_html=True)
    #AgGrid(power_pred_df)
    kpi_lines = alt.Chart(power_pred_df).mark_line().mark_line().encode(x='hours',
                                                                        y=alt.Y(f'{pred_linechart_kpi}_pred_mean').title("Mega Watts"),
                                                                        color=alt.Color('mean_label',legend=alt.Legend(
                                                                                                    orient='none',
                                                                                                    legendX=490, legendY=0,
                                                                                                    direction='horizontal',
                                                                                                    titleAnchor='middle'
                                                                                                    )
                                                                                            )
                                                                                )
    kpi_lines.encoding.x.scale = alt.Scale(domain=[0, 168])
                           

    pred_band = (alt.Chart(power_pred_df).mark_area(opacity=0.5, color= 'pink').encode(x='hours', 
                                                        y=alt.Y(f'{pred_linechart_kpi}_pred_upper:Q').title(""),
                                                        y2=alt.Y2(f'{pred_linechart_kpi}_pred_lower:Q').title(""),
                                                        color=alt.Color('stddev_label',legend=alt.Legend(
                                                                                                    title='Legend',
                                                                                                    orient='none',
                                                                                                    legendX=490, legendY=0,
                                                                                                    direction='horizontal',
                                                                                                    titleAnchor='middle'
                                                                                                    )
                                                                                            )
                                            )
    )
    pred_band.encoding.x.scale = alt.Scale(domain=[0, 168])

    st.altair_chart((kpi_lines+pred_band), use_container_width=True)
    return power_pred_df


def show_error_metrics(pred_df, kpi):

    actual_val = pred_df[kpi]
    pred_val = pred_df[f'{kpi}_pred_mean']
    n = len(pred_df)
    print(n)
    mse = np.sum(np.square(actual_val - pred_val))/n
    mape = (np.sum(np.abs(np.divide((actual_val-pred_val), actual_val)))/n)*100 
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.markdown(f'<h2> MAPE </h2>', unsafe_allow_html=True)
        st.markdown(f'<h3 style="text-align:center"> {mape:.3f}</h3>', unsafe_allow_html=True)
    with metrics_col2:
        st.markdown(f'<h2> MSE </h2>', unsafe_allow_html=True)
        st.markdown(f'<h3 style="text-align:center"> {mse:.3f}</h3>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Telecom Dashboard",
        page_icon="✅",
        layout="wide",
        
    )
    st.html("../styles.html")
    st.session_state.llm='llama3'
    setup_llms()
    setup_llm_chains()
    st.session_state.rerun_dashboard = True
    kpi_list = ['total_energy_output', 'reservoir_level', 'water_flow_rate', 'co2_emissions']
    # read csv from a github repo
    #dataset_url = "../data/dashboard_data.csv"
    st.session_state.dataset_url = "../data/dashboard/dashboard_monitoring_data.csv"
    st.session_state.full_data_df = get_data()
    plant_names = st.session_state.full_data_df['name'].unique()

    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        param_form_container = st.container(height=500)
        chat_container = st.container(height=380)
        with param_form_container:
           build_data_filter_window()
        with chat_container:
            build_chat_window_assistant()
    with col2:
        pred_df = None
        pred_plot_container = st.container(height=700, border=True)
        pred_stats_container = st.container(height=180)
        pred_linechart_kpi = 'total_energy_output'
        with pred_plot_container:
            with st.form("Evaluate on "):
                eval_col1, eval_col2, eval_col3 = st.columns([0.1, 0.3, 0.4])
                with eval_col2:
                    plant_name = st.selectbox('Select Plant', plant_names)
                with eval_col3:
                    predict_button = st.form_submit_button("Run Predition")
            if predict_button:
                pred_df = plot_kpi_prediction_data(plant_name, pred_linechart_kpi)
                with pred_stats_container:    
                    show_error_metrics(pred_df, pred_linechart_kpi)
if __name__=='__main__':
    main()


