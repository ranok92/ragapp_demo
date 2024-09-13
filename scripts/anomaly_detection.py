
import os 
#--- streamlit and other UI imports 
import streamlit as st  # 🎈 data web app development
from streamlit_folium import st_folium
from streamlit_extras.stylable_container import stylable_container
import altair as alt
import folium
from folium.plugins import Realtime, MarkerCluster
from streamlit_folium import st_folium
from folium import JsCode
import statistics
#--- llm imports 
import re 
from langchain_community.llms import Ollama
from langchain import LLMChain, PromptTemplate

#---- local imports ----
from utils.utils import *
from utils.dashboard_utils import *
from prompts.prompt_template import *
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import datetime

with open('../assets/openai_api_key.txt', 'r') as f:
    key = f.read()
os.environ["OPENAI_API_KEY"]=key

REFRESH_TIMER = 2
@st.experimental_fragment(run_every=REFRESH_TIMER)
def get_data_anomaly() -> pd.DataFrame:
    st.session_state.cur_data_df = pd.read_csv(st.session_state.cur_dataset_url,  index_col=False)

    t = st.session_state.cur_data_df['timestamp'].iloc[0]

    if 'timestamps' not in st.session_state.keys():
        st.session_state.timestamps = []
    
    if t not in st.session_state.timestamps:
        st.session_state.timestamps.append(t)
        if 'cumm_data_df' not in st.session_state.keys():
            st.session_state.cumm_data_df = st.session_state.cur_data_df
        else:
            st.session_state.cumm_data_df = pd.concat([st.session_state.cumm_data_df, st.session_state.cur_data_df], ignore_index=True)


def get_data_full_anomaly():
    st.session_state.full_data_df = pd.read_csv(st.session_state.dataset_url)


#--- setup llms and llm chains ---
def setup_llms_anomaly():

    st.session_state.llm_model_chat = Ollama(model='llama3.1', system='You are a helpful question answering bot.')
    st.session_state.llm_model_instruct = Ollama(model='llama3.1', format='json', system="You are an LLM who is logical and is excellent at following instructions.")
    # st.session_state.llm_dashboard_assistant = Ollama(model='llama3.1', format='json', system="You are a bot who specializes on reading tabular data, summarizing them and providing insights.")

    #OPENAI
    st.session_state.llm_dashboard_assistant = ChatOpenAI(
                                                model="gpt-4o-mini" ,
                                                temperature=0,
                                                max_retries=2,
                                            )

def setup_llm_chains_anomaly():

    #build the conversation chain
    conv_prompt = PromptTemplate(input_variables=['input', 'history'], template=CONV_PROMPT_TEMPLATE)
    st.session_state.conv_chain = LLMChain(llm=st.session_state.llm_model_chat, 
                                           prompt=conv_prompt, 
                                           output_key='answer')
    
    #build the rephrase chain 
    rephrase_prompt = PromptTemplate(input_variables=['input', 'chat_history'], template=RETRIEVE_REPHRASE_PROMPT)
    st.session_state.rephrase_chain = LLMChain(llm=st.session_state.llm_dashboard_assistant, 
                                               prompt=rephrase_prompt)

    #build the router chain
    router_prompt = PromptTemplate(
        input_variables=["input"], template=ROUTER_PROMPT_TEMPLATE_BASIC
    )
    st.session_state.router_chain_anomaly = LLMChain(llm=st.session_state.llm_model_instruct, prompt=router_prompt, output_key='answer')

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

#----------------------

#----- Streamlit page building functions ------------
@st.experimental_fragment(run_every=REFRESH_TIMER)
def write_latest_update_time():
    st.markdown(f"<h2 style='text-align: center; color: #453030;'> Outage Tracker </h2> <p style='text-align: right'> Last Updated : {st.session_state.cur_data_df['datetime'][0]} ", unsafe_allow_html=True)


def draw_realtime_map():

    m = folium.Map(location=[st.session_state.full_data_df['latitude'].mean(), st.session_state.full_data_df['longitude'].mean()], zoom_start=10)
    source_anomaly= 'http://localhost:8000/anomalous.geojson'
    container = MarkerCluster(icon_create_function=icon_create_function).add_to(m)
    pt_layer_func = JsCode('''(f, latlng) => { 
                                var rad = f.properties.people_affected/20
                                var popup_options = {className:'popupclass'}
                                var popup_msg = '<p> Area code:' + f.properties.postal_code + "<br>" + 'Time :' + f.properties.start_time + "<br>" +  'Reason :' + f.properties.outage_reason + "<br>" + 'People affected :' + f.properties.people_affected + "</p>"
                                return L.circleMarker(latlng, {radius: 10, fillOpacity: 0.4, color: '#cf1313', fillColor: '#cf1313', interactive: true}).bindPopup(popup_msg, popup_options); }
                           
                           ''')
    realtime_layer_anomaly = Realtime(
        source_anomaly,  # Local URL to the GeoJSON file
        start=True,  # Automatically start refreshing
        get_feature_id=JsCode("(f) => { return f.properties.objectID}"),
        remove_missing=True,
        container=container,
        point_to_layer=pt_layer_func,
        interval=1000
    )
    realtime_layer_anomaly.add_to(m)
    st_folium(m, height=500, use_container_width=True)


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

def get_session_anomaly_chat_history():
    chat_list = st.session_state.messages_anomaly 
    chat_history = []
    for conv in chat_list:
        if conv['speaker']=="user":
            chat_history.append(HumanMessage(content=conv['content']))
        if conv['speaker']=='AI':
            chat_history.append(AIMessage(content=conv['content']))
    return chat_history

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
    router_samples = 1
    router_resp_list = []
    for i in range(router_samples):

        router_resp = st.session_state.router_chain_anomaly.invoke({'input': resp_string})
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
        print("Response from TABLE :", result)
        anno_result = result['text']
        st.session_state.response_context = ""  
 
    
    #save the query in the chat history
    st.session_state.messages_anomaly.append({"speaker" : "user", "content": query_text})
    
    # rel_sources = [doc.metadata['source'] for doc in docs]
    # rel_pages = [doc.metadata['page'] for doc in docs]
    # rel_data_resp = f'\n Relevant information can be found in the following documents : {" ".join(rel_sources)}'

    st.session_state.messages_anomaly.append({"speaker" : "AI",
                                    "content": re.sub('\$','\\$',anno_result)
})


@st.experimental_fragment(run_every=REFRESH_TIMER)
def plot_outage_occurance_linechart():
    #aggregate the anomalies by datetime
    st.markdown("<h2 style='text-align: center; color: #453030;'> Outages Registered Over Time</h2>", unsafe_allow_html=True)

    part_data_df = st.session_state.cumm_data_df
 
    part_data = part_data_df.groupby(['datetime', 'outage_reason']).agg(counts=('outage_reason', 'count')).reset_index()
    
    part_data.rename(columns={'outage_reason':'Outage Category'}, inplace=True)
    part_data_df_long = part_data.melt(ignore_index=False, var_name='Anomaly Type')
    part_data_long_w_index = part_data_df_long.reset_index()
    ano_chart = alt.Chart(part_data).mark_area(opacity=0.5).encode(
            x='datetime:T',
            y='counts:Q',
            color='Outage Category:N'
    )
    st.altair_chart(ano_chart, use_container_width=True)

@st.experimental_fragment(run_every=REFRESH_TIMER)
def write_outages():
    t = st.session_state.timestamps[-1]
    st.markdown("<h2 style='text-align: center; color: #453030; padding: 1rem 0px'> Outage Reason </h2>", unsafe_allow_html=True)
    row1 = st.container()
    row2 = st.container()
    row3 = st.container()
    outage_counts_df = st.session_state.cumm_data_df.groupby('outage_reason').agg(outage_counts=('outage_reason', 'count')).reset_index()
    with row1:
        ano_col1, ano_col2  = st.columns(2)
        with ano_col1:
            st.html(f'<span class="anomaly_counter"></span>')

            env_factors = outage_counts_df[outage_counts_df['outage_reason']=='Environmental Factors']['outage_counts'].iloc[0]
            st.markdown("<h5 style='text-align: center; color: black;'>Env. Factors</h5>", unsafe_allow_html=True)
            st.write(f"<h6> {env_factors} </h6>", unsafe_allow_html=True)
        with ano_col2:
            st.html(f'<span class="anomaly_counter"></span>')
            eqp_fail = outage_counts_df[outage_counts_df['outage_reason']=='Equipment Failure']['outage_counts'].iloc[0]
            st.markdown("<h5 style='text-align: center; color: black;'>Equipment Failure</h5>", unsafe_allow_html=True)
            st.write(f"<h6> {eqp_fail} </h6>", unsafe_allow_html=True)
    with row2:

        ano_col3, ano_col4 = st.columns(2)
        with ano_col3:
            st.html(f'<span class="anomaly_counter"></span>')

            ext_factors = outage_counts_df[outage_counts_df['outage_reason']=='External Factors']['outage_counts'].iloc[0]
            st.markdown("<h5 style='text-align: center; color: black;'>Ext. Factors</h5>", unsafe_allow_html=True)
            st.write(f"<h6> {ext_factors} </h6>", unsafe_allow_html=True)
        with ano_col4:
            st.html(f'<span class="anomaly_counter"></span>')

            nat_cause = outage_counts_df[outage_counts_df['outage_reason']=='Natural Cause']['outage_counts'].iloc[0]
            st.markdown("<h5 style='text-align: center; color: black;'>Nat. Causes</h5>", unsafe_allow_html=True)
            st.write(f"<h6> {nat_cause} </h6>", unsafe_allow_html=True)

    with row3:
        ano_col5, ano_col6 = st.columns(2)
        with ano_col5:
            st.html(f'<span class="anomaly_counter"></span>')

            sys_repair = outage_counts_df[outage_counts_df['outage_reason']=='Power System Repair']['outage_counts'].iloc[0]
            st.markdown("<h5 style='text-align: center; color: black;'>Repair</h5>", unsafe_allow_html=True)
            st.write(f"<h6> {sys_repair} </h6>", unsafe_allow_html=True)
        with ano_col6:

            st.html(f'<span class="anomaly_counter"></span>')
            sys_improvement = outage_counts_df[outage_counts_df['outage_reason']=='System Improvement']['outage_counts'].iloc[0]
            st.markdown("<h5 style='text-align: center; color: black;'>Improvement</h5>", unsafe_allow_html=True)
            st.write(f"<h6> {sys_improvement} </h6>", unsafe_allow_html=True)

#---- Methods not being used for now ----

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
# ----------------------------------------------------

def main():
    #create the header container
    st.session_state.dataset_url = "../data/dashboard/outage_monitoring_data.csv"
    st.session_state.cur_dataset_url = "../data/dashboard/outage_monitoring_data_per_hr.csv"
    get_data_anomaly()
    get_data_full_anomaly()

    setup_llms_anomaly()
    setup_llm_chains_anomaly()
    st.html("../css/timeseries_page_styles.html")

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
            st.markdown(f"<h2 style='text-align: center; color: #453030;'> Outage Tracker </h2> <p style='text-align: right'> Last Updated : {st.session_state.cur_data_df['datetime'][0]} ", unsafe_allow_html=True)

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

if __name__=='__main__':
    main()

