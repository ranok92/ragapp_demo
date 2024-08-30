
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
from langchain_core.messages import HumanMessage, AIMessage

FORECASTING_FEATURES = [
    # Meteorological Data
    "solar_irradiance",           # Solar energy received per unit area (W/m²)
    "temperature",                # Ambient and module temperature
    "cloud_cover",                # Cloud presence and density
    "wind_speed",                 # Wind speed
    "humidity",                   # Atmospheric humidity
    "precipitation",              # Rain or snow affecting solar panels

    # Temporal Features
    "time_of_day",                # Hour of the day
    "day_of_year",                # Seasonal variations (day of the year or month)
    "day_of_week",                # Day of the week
    "historical_power_output",    # Previous power generation data

    # Geographical and Location-Based Features
    "latitude",                   # Latitude of the solar farm
    "longitude",                  # Longitude of the solar farm
    "altitude",                   # Altitude of the solar farm
    "panel_orientation",          # Orientation of the solar panels
    "panel_tilt",                 # Tilt angle of the solar panels
    "shading_obstacles",          # Nearby obstacles causing shading

    # Operational Features
    "panel_type",                 # Type of solar panels (e.g., monocrystalline, polycrystalline)
    "panel_age",                  # Age of the solar panels
    "inverter_efficiency",        # Efficiency of inverters
    "maintenance_records",        # Maintenance and cleaning records
    "battery_storage_levels",     # Current levels and usage patterns of battery storage

    # External Influences
    "grid_demand",                # Grid demand impacting solar farm operations
    "curtailment",                # Curtailment orders

    # Derived and Engineered Features
    "clear_sky_solar_irradiance", # Estimated maximum irradiance under clear sky
    "lagged_features",            # Previous time steps' values for temporal dependencies
    "moving_averages",            # Moving averages or rolling windows

    # Environmental Data
    "aerosol_levels",             # Dust, pollution, and other aerosols in the atmosphere
    "albedo"                      # Reflectivity of the surrounding surface
]

FORECASTING_MODELS = [
    # Recurrent Neural Networks (RNNs)
    "Basic RNNs",                  # Good for handling sequential data with short-term dependencies
    
    # Long Short-Term Memory Networks (LSTMs)
    "LSTMs",                       # Excellent for capturing long-term dependencies in time series
    
    # Gated Recurrent Units (GRUs)
    "GRUs",                        # Similar to LSTMs but with a simpler architecture and faster training
    
    # Convolutional Neural Networks (CNNs)
    "CNNs for Time Series",        # Effective for detecting local patterns, often combined with RNNs or LSTMs
    
    # Temporal Convolutional Networks (TCNs)
    "TCNs",                        # Designed specifically for sequential data, capturing long-range dependencies
    
    # Transformer Models
    "Transformers",                # Uses self-attention mechanisms to capture long-range dependencies
    
    # Hybrid Models
    "CNN-LSTM",                    # Combines CNNs and LSTMs for feature extraction and sequential modeling
    "Seq2Seq (Sequence to Sequence)", # Maps an input sequence to an output sequence, suitable for multi-step forecasting
    "Attention Mechanisms",        # Added to RNNs or LSTMs to focus on specific parts of the sequence
    
    # Feedforward Neural Networks (FFNNs)
    "FFNNs with Feature Engineering", # Uses engineered features from the time series for basic forecasting tasks
    
    # DeepAR (Amazon)
    "DeepAR",                      # Combines autoregressive models with deep learning for probabilistic forecasting
    
    # N-BEATS
    "N-BEATS (Neural Basis Expansion Analysis Time Series)" # Deep learning model for time series forecasting
]

FORECASTING_OPTIMIZERS = [
    # Gradient Descent-based Optimizers
    "SGD",                        # Stochastic Gradient Descent - Basic optimizer with optional momentum.
    "SGD with Momentum",           # SGD enhanced with momentum to accelerate convergence.
    
    # Adaptive Learning Rate Optimizers
    "Adam",                       # Adaptive Moment Estimation - Combines the benefits of AdaGrad and RMSprop.
    "RMSprop",                    # Root Mean Square Propagation - Adjusts the learning rate based on recent gradients.
    "Adagrad",                    # Adaptive Gradient Algorithm - Adapts learning rates based on individual parameters.
    "Adadelta",                   # Extension of Adagrad that seeks to address its learning rate decay issues.
    "AdamW",                      # Adam with Weight Decay - Like Adam, but includes decoupled weight decay for better regularization.
    "Nadam",                      # Nesterov-accelerated Adaptive Moment Estimation - Adam combined with Nesterov momentum.
    "AdaMax",                     # A variant of Adam based on the infinity norm.
    
    # Second-Order Methods
    "L-BFGS",                     # Limited-memory Broyden–Fletcher–Goldfarb–Shanno - A quasi-Newton method that approximates the second-order derivative (Hessian).
    
    # Other Optimizers
    "FTRL",                       # Follow-the-Regularized-Leader - Used in large-scale linear models.
    "Yogi",                       # An optimizer like Adam, but more robust to noisy gradients and less sensitive to hyperparameter settings.
    "Rprop",                      # Resilient Backpropagation - Adjusts the step size for each weight independently.
    "AMSGrad",                    # A variant of Adam that seeks to improve convergence by enforcing a non-increasing step size.
    "SWATS",                      # Switches from Adam to SGD when necessary to potentially improve generalization.
]

FORECASTING_LOSS_FUNCTIONS = [
    # Regression Loss Functions
    "Mean Squared Error (MSE)",             # Penalizes the square of the difference between predicted and actual values, sensitive to outliers.
    "Mean Absolute Error (MAE)",            # Penalizes the absolute difference between predicted and actual values, more robust to outliers.
    "Huber Loss",                           # Combines MSE and MAE, less sensitive to outliers than MSE, but differentiable everywhere.
    "Mean Absolute Percentage Error (MAPE)",# Expresses the error as a percentage of the actual values, useful for interpretability.
    "Root Mean Squared Error (RMSE)",       # The square root of MSE, provides error in the same units as the output.
    "Quantile Loss",                        # Used in quantile regression to predict a specific quantile, useful for uncertainty estimation.
    "Log-Cosh Loss",                        # The logarithm of the hyperbolic cosine of the prediction error, smooths out large differences.
    
    # Distribution-based Loss Functions
    "Negative Log Likelihood",              # Measures the likelihood of the observed data under the predicted probability distribution.
    "Pinball Loss",                         # A variation of Quantile Loss, used for interval predictions.

    # Custom Loss Functions
    "Asymmetric Loss",                      # Custom loss where overestimations and underestimations are penalized differently.
    "Smoothed L1 Loss",                     # A variation of Huber Loss, smooths transitions between L1 and L2 loss behaviors.

    # Specific to Forecasting
    "Symmetric Mean Absolute Percentage Error (sMAPE)", # A variant of MAPE that treats over- and under-forecasts symmetrically.
    "Mean Squared Logarithmic Error (MSLE)", # Penalizes the square of the logarithmic difference between predicted and actual values, reduces impact of large errors.

    # Probabilistic Loss Functions
    "Gaussian Negative Log-Likelihood",     # Assumes the data follows a Gaussian distribution and calculates the negative log-likelihood.
    "Poisson Loss",                         # Suitable for count data, assumes the data follows a Poisson distribution.
    "CRPS (Continuous Ranked Probability Score)", # Measures the accuracy of probabilistic forecasts.
]


def get_data_forecast():
    return pd.read_csv(st.session_state.forecast_dataset_url)

def get_session_forecast_chat_history():
    chat_list = st.session_state.messages_forecast 
    chat_history = []
    for conv in chat_list:
        if conv['speaker']=="user":
            chat_history.append(HumanMessage(content=conv['content']))
        if conv['speaker']=='AI':
            chat_history.append(AIMessage(content=conv['content']))
    return chat_history


def setup_llms_forecast():
    st.session_state.llm_dashboard_assistant = Ollama(model='llama3.1', 
                            system="You are a bot who is an expert on timeseries model prediction.")

def setup_llm_chains_forecast():

    #build the conversation chain
    pred_assistant_prompt = PromptTemplate(input_variables=['input', 'history'], template=PRED_ASSISTANT_PROMPT_TEMPLATE)
    st.session_state.assistant_chain = LLMChain(llm=st.session_state.llm_dashboard_assistant, prompt=pred_assistant_prompt, output_key='answer')
    
def query_chain_forecast():
    form_info = None
    if 'timeseries_form_info' in st.session_state.keys():
        form_info = st.session_state.timeseries_form_info
    
    input_query = st.session_state.current_forecast_input
    st.session_state.messages_forecast.append({"speaker" : "user", "content": input_query})
    resp = st.session_state.assistant_chain.invoke({'input':input_query, 
                                                    'history': get_session_forecast_chat_history(),
                                                    'user_param_choices' : form_info})
    
    print("RESP : ", resp)
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
    st.markdown(f'<h3 style="color:black ;text-align:center">Model Definition </h2>', unsafe_allow_html=True)
    param_select_form = st.form('Select params', border=False)
    with param_select_form:
        #form_col1, form_col2 = st.columns(2)
        
        features_selected = st.multiselect("Features to include", 
                                        FORECASTING_FEATURES,
                                        )
        model_selected = st.selectbox("Pick a model", 
                                    FORECASTING_MODELS)
            
        add_normalization = st.toggle('Add normalization')
        add_dropout = st.toggle('Add dropout')

        loss_func_selected = st.selectbox("Pick a loss function", 
                                            FORECASTING_LOSS_FUNCTIONS)
        optimizer_selected = st.selectbox("Pick an optimizer", 
                                        FORECASTING_OPTIMIZERS)
        training_epochs = st.text_input('Training Epochs', 1000)
        learning_rate = st.text_input("Learning rate", 0.001)
     
    
        get_form_data = st.form_submit_button("Set Params")
        if get_form_data:
            st.session_state.timeseries_form_info = {}
            st.session_state.timeseries_form_info['features_selected'] = features_selected
            st.session_state.timeseries_form_info['model_selected'] = model_selected
            st.session_state.timeseries_form_info['normalization'] = add_normalization
            st.session_state.timeseries_form_info['dropout'] = add_dropout
            st.session_state.timeseries_form_info['loss_func_selected'] = loss_func_selected 
            st.session_state.timeseries_form_info['optimizer_selected'] = optimizer_selected 
            st.session_state.timeseries_form_info['learning_rate_selected'] = learning_rate

   
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
    st.html("../css/timeseries_page_styles.html")
    st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>', unsafe_allow_html=True)    
    setup_llms_forecast()
    setup_llm_chains_forecast()
    print("running forecast tab")

    # read csv from a github repo
    st.session_state.forecast_dataset_url = "../data/dashboard/solar_powerplant_forecasting_data.csv"
    st.session_state.full_forecast_data_df = get_data_forecast()
    plant_names = st.session_state.full_forecast_data_df['name'].unique()
    pred_linechart_kpi = 'total_energy_output'
    pred_df = None
    #design the UI
    with stylable_container(
        key='page_header',
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


