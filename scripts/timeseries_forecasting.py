
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import streamlit as st  # 🎈 data web app development
from streamlit_extras.stylable_container import stylable_container
from utils.utils import *
from utils.dashboard_utils import *
import json
import altair as alt 

from prompts.prompt_template import *
from langchain_community.llms import Ollama
from langchain import LLMChain, PromptTemplate

def get_data_forecast():
    return pd.read_csv(st.session_state.forecast_dataset_url)


def setup_llms_forecast():
    st.session_state.llm_dashboard_assistant = Ollama(model='llama3.1', 
                            system="You are a bot who is an expert on timeseries model prediction.")

def setup_llm_chains_forecast():

    #build the conversation chain
    pred_assistant_prompt = PromptTemplate(input_variables=['input', 'history'], template=PRED_ASSISTANT_PROMPT_TEMPLATE)
    st.session_state.assistant_chain = LLMChain(llm=st.session_state.llm_dashboard_assistant, prompt=pred_assistant_prompt, output_key='answer')
    
def query_chain_forecast():
    input_query = st.session_state.current_input
    st.session_state.messages_forecast.append({"speaker" : "user", "content": input_query})
    resp = st.session_state.assistant_chain.invoke({'input':input_query})
    # rel_sources = [doc.metadata['source'] for doc in docs]
    # rel_pages = [doc.metadata['page'] for doc in docs]
    # rel_data_resp = f'\n Relevant information can be found in the following documents : {" ".join(rel_sources)}'
    st.session_state.messages_forecast.append({"speaker" : "AI",
                                    "content": resp['answer']})



def build_chat_window_forecast_assistant():
    #st.markdown(f'<h3 style="color:black; text-align:center">Forecasting Assistant</h3>', unsafe_allow_html=True)

    if "messages_forecast" not in st.session_state:
        st.session_state.messages_forecast = []
    st.chat_input(placeholder = 'Enter query here ...', 
                on_submit=query_chain_forecast,
                key='current_forecast_input')
    chat_row_assistant = st.empty()
    #context_row = st.empty()
    with chat_row_assistant.container(height=230, border=True):
        #display the chat history so far
        for msg in st.session_state.messages_forecast:
            st.chat_message(msg['speaker']).markdown(msg['content'])


def build_param_selection_form():
    st.markdown(f'<h3 style="color:black ;text-align:center">Param Selection</h2>', unsafe_allow_html=True)
    param_select_form = st.form('Select params', border=False)
    with param_select_form:
        #form_col1, form_col2 = st.columns(2)
        
        features_selected = st.multiselect("Features to include", 
                                        ['Feature1', 'Feature2', 'Feature3', 'Feature4'],
                                        'Feature1')
        model_selected = st.selectbox("Pick a model", 
                                    ['Model 1', 'Model 2', 'Model 3'])
            
        add_normalization = st.toggle('Add normalization')
        add_dropout = st.toggle('Add dropout')

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
    plant_power_data_predict_mean = st.session_state.full_forecast_data_df[st.session_state.full_forecast_data_df['name']==plant_name][f'{pred_linechart_kpi}_predict_mean']
    plant_power_data_predict_std = st.session_state.full_forecast_data_df[st.session_state.full_forecast_data_df['name']==plant_name][f'{pred_linechart_kpi}_predict_std']
    kpi_data = list(st.session_state.full_forecast_data_df[st.session_state.full_forecast_data_df['name']==plant_name][f'{pred_linechart_kpi}'])[0:t+1]
    #kpi_data = []

    power_pred_df = pd.DataFrame()
    power_pred_df['hours'] = np.arange(timesteps+1)

    #current data
    kpi_data.extend([float("NaN")]*(timesteps-t))
    power_pred_df[f'{pred_linechart_kpi}'] = list(st.session_state.full_forecast_data_df[st.session_state.full_forecast_data_df['name']==plant_name][f'{pred_linechart_kpi}'])
    #pred mean
    pred_mean_nan = [float("NaN")]*t
    pred_mean_future = plant_power_data_predict_mean[t:]
    pred_mean_nan.extend(pred_mean_future)

    #pred_std
    pred_std_nan = [float("NaN")]*t
    pred_std_future = plant_power_data_predict_std[t:]
    pred_std_nan.extend(pred_std_future)

    #add noise to mean to emulate different perdictions 
    pred_mean_nan_noisy = pred_mean_nan+ np.random.uniform(low=np.zeros(len(pred_mean_nan)), 
                                                                high=np.array(pred_mean_nan)/5)
    #add cols to df
    power_pred_df[f'{pred_linechart_kpi}_pred_mean'] = pred_mean_nan_noisy
    #+np.random.rand(168)*2
    #power_pred_df[f'']

    multiplier = power_pred_df[f'{pred_linechart_kpi}_pred_mean'].mean()/5
    #multiplier = 1
    power_pred_df[f'{pred_linechart_kpi}_pred_upper'] = power_pred_df[f'{pred_linechart_kpi}_pred_mean']+np.array(pred_std_nan)+np.random.rand(168)*multiplier+multiplier/5
    power_pred_df[f'{pred_linechart_kpi}_pred_lower'] = power_pred_df[f'{pred_linechart_kpi}_pred_mean']-np.array(pred_std_nan)-np.random.rand(168)*multiplier-multiplier/5
    power_pred_df['actual_kpi_label'] = (timesteps+1)*['actual value']
    power_pred_df['mean_label'] = (timesteps+1)*['predicted mean']
    power_pred_df['stddev_label'] = (timesteps+1)*['predicted std deviation']

    line_plot_df = power_pred_df[['hours', 
                                  f'{pred_linechart_kpi}',
                                    f'{pred_linechart_kpi}_pred_mean']]
    
    line_plot_df.rename(columns={f'{pred_linechart_kpi}': 'Actual Value', 
                                    f'{pred_linechart_kpi}_pred_mean': 'Predicted Value'},
                                    inplace=True)

    line_plot_df = line_plot_df.melt(id_vars=['hours'],
                                     value_vars=['Actual Value', 'Predicted Value'],
                                        var_name='Entity', value_name='m_watts', ignore_index=True)
    line_plot_df['mean_label'] = (timesteps+1)*(len(line_plot_df))

    #AgGrid(power_pred_df)
    kpi_lines = alt.Chart(line_plot_df, height=600).mark_line().encode(x=alt.X('hours'),
                                                                        y=alt.Y('m_watts', axis=alt.Axis(tickCount=30)).title("Mega Watts"),
                                                                        strokeDash='Entity',
                                                                        # color=alt.Color('mean_label',legend=alt.Legend(
                                                                        #                             orient='none',
                                                                        #                             legendX=450, legendY=0,
                                                                        #                             direction='horizontal',
                                                                        #                             titleAnchor='middle'
                                                                        #                             )
                                                                        #                     )
                                                                                )
    
    
    kpi_lines.encoding.x.scale = alt.Scale(domain=[0, 168])
                       

    pred_band = (alt.Chart(power_pred_df).mark_area(opacity=0.3).encode(x='hours', 
                                                                        y=alt.Y(f'{pred_linechart_kpi}_pred_upper:Q').title(""),
                                                                        y2=alt.Y2(f'{pred_linechart_kpi}_pred_lower:Q').title(""),
                                                                        # color=alt.Color('stddev_label',legend=alt.Legend(
                                                                        #                                             title='Legend',
                                                                        #                                             orient='none',
                                                                        #                                             legendX=650, legendY=-30,
                                                                        #                                             direction='horizontal',
                                                                        #                                             titleAnchor='end'
                                                                        #                                             )
                                                                        #                                     )
                                                                                 )
    )
    pred_band.encoding.x.scale = alt.Scale(domain=[0, 168])
    full_chart = kpi_lines+pred_band
    full_chart.configure_view(cornerRadius=100)

    st.altair_chart((full_chart), use_container_width=True)
    return power_pred_df


def show_error_metrics(pred_df, kpi):

    actual_val = pred_df[kpi]
    pred_val = pred_df[f'{kpi}_pred_mean']
    n = len(pred_df)
    rmse = np.sqrt(np.sum(np.square(actual_val - pred_val))/n)
    mape = (np.sum(np.abs(np.divide((actual_val-pred_val), actual_val)))/n)*100 
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        with stylable_container(
            key='metric1',
            css_styles='''
                {
                    width: 90%;
                    margin-bottom: 0px;
                    background: white;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-radius: 10px;
                    padding-bottom: 5px;
                    text-align:center;
                }
                '''
        ):
            st.markdown(f'<h4> MAPE </h4>', unsafe_allow_html=True)
            st.markdown(f'<h3 style="text-align:center"> {mape:.3f}</h3>', unsafe_allow_html=True)
    with metrics_col2:
        with stylable_container(
            key='metric2',
            css_styles='''
                {
                    width: 90%;
                    margin-bottom: 0px;
                    background: white;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-radius: 10px;
                    pdding-bottom: 5px;
                    text-align:center;

                }
                '''
        ):
            st.markdown(f'<h4> RMSE </h4>', unsafe_allow_html=True)
            st.markdown(f'<h3 style="text-align:center"> {rmse:.3f}</h3>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Telecom Dashboard",
        page_icon="✅",
        layout="wide",
        
    )
    st.html("../timeseries_page_styles.html")
    st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>', unsafe_allow_html=True)    
    setup_llms_forecast()
    setup_llm_chains_forecast()
    # read csv from a github repo
    st.session_state.forecast_dataset_url = "../data/dashboard/dashboard_monitoring_data.csv"
    st.session_state.full_forecast_data_df = get_data_forecast()
    plant_names = st.session_state.full_forecast_data_df['name'].unique()
    pred_linechart_kpi = 'total_energy_output'
    pred_df = None
    #design the UI
    with stylable_container(
        key='page_header',
        css_styles='''
        {
        width: 90%;
        justify-content: space-around;
        border-radius: 15px;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding-left:30px;
        padding-bottom:20px
        }
''',
    ):
        st.markdown(f'<h1 style="color: white;"> Forecast Dashboard </h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.27, 0.63, 0.1])
    with col1:
        param_form_container = st.container(height=800, border=True)
        run_eval_container = st.container(height=250, border=True)
    with col2:
        pred_stats_container = st.container(height=300, border=False)   
        
        pred_plot_container = st.container(height=750, border=True)
        with pred_plot_container:
            st.markdown("<h3 style='text-align: center; color: black;'> Forecast Plot </h3>", unsafe_allow_html=True)
    with col3:
        chat_container = st.container(height=150, border=False)
        with chat_container:
            with st.popover(":headphones:", help='Model Consultant'):
                build_chat_window_forecast_assistant()
        blank_container = st.container(height=900, border=False)

    with col1:
        with param_form_container:
           build_param_selection_form()
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
if __name__=='__main__':
    main()


