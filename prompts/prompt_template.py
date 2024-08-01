from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ROUTER_PROMPT_TEMPLATE = "Given an input, decide whether responding to it would require fact based question answering capabilites or it is a part of a general conversation.\
                    Here is the input \n\n {input}.\n\n Say Yes or no. Yes, if you think it would require fact based question answering capabilities, no, otherwise. Explain your answer.\n \
                    Respond with a json with two keys. 'response' and 'explaination'. 'response' should either be 'yes' or 'no."


ROUTER_PROMPT_TEMPLATE_2 = "Given an input, decide whether responding to it would require fact based question answering capabilitesl, a part of a general conversation or require.\
                    assistance with creative writing. \n Here is the input \n\n {input}.\n\n Repond with one word: 'qa', 'conv' or 'writing'. 'qa', if you think it \
                    would require fact based question answering capabilities, 'conv', if you think it is a part of a regular conversation and 'writing' if you understand that \
                    the input is requesting help with creative writing. \n \
                    Explain your answer.\n \
                    Respond with a json with two keys. 'response' and 'explaination'. 'response' should either be 'qa', 'conv' or 'writing'."



ROUTER_PROMPT_TEMPLATE_3 = "Given an input, decide whether responding to it would require fact based question answering capabilitesl, a part of a general conversation, \
                    need to consult a table or require assistance with creative writing. \
                    \n Here is the input \n\n {input}.\n\n Repond with one word: 'qa', 'conv','table', or 'writing'. \
                    'qa', if you think it would require fact based question answering capabilities, \
                    'conv', if you think it is a part of a regular conversation, \
                    'table' if you think it would need to access tabular data, and \
                    'writing' if you understand that \
                    the input is requesting help with creative writing. \n \
                    Explain your answer.\n \
                    Respond with a json with two keys. 'response' and 'explaination'. 'response' should either be 'qa', 'conv', 'table' or 'writing'."


CONV_PROMPT_TEMPLATE = "You are a helpful bot who can hold a polite conversation with a fellow human. \
                        You will be provided with a history of messages. Based on that you need to form a final \
                        response. Try not to be too wordy.\n\n \n\n \
                        The chat history: {chat_history}\n \
                        Human's last chat: {input}"


RAG_PROMPT_TEMPLATE =  "Answer the user's questions based on the context provided.\n \
         If you don't know the answer just say you dont know. Do not try to come up with something. \n \
         Keep your answer brief and to the point. \n\n \
         The context: {context} \n \
         Human's question: {input}"


TABLE_SUMMARIZER_TEMPLATE = '''
You are a bot who specializes on reading tabular data and summarizing the contents. \n
The table data you will read belongs to a power grid system. Try to provide 
suggestions as to why the anomaly occured along with the summary. \n\n Just provide your thoughts. 
No need to ask for feedback. Always respond in third person.\\
Respond with a dictionary with two keys: 'summary' and 'thoughts'
====
Example 1:

Table data:
    name	type	watershed	capacity (mw)	head (m)	units	water_flow_rate	reservoir_level	total_energy_output	co2_emissions	anomaly_total_energy_output	anomaly_water_flow_rate	anomaly_co2_emissions	anomaly_reservoir_level	anomaly
    18	Beauharnois	Run of river	Saint Lawrence	1906	24.39	38.0	3793.0	41.0	312.0	15.0	0	1	0	0	1
    18	Bersimis-2	Run of river	Betsiamites	869	115.83	5.0	1816.0	39.0	109.0	21.5	1	0	0	0	1
    18	Brisay	Reservoir	Caniapiscau	469	37.5	2.0	2384.0	52.0	58.0	8.0	1	0	0	0	1
    18	Chute-Hemmings	Run of river	Saint-François	29	14.64	6.0	397.0	37.0	5.0	4.0	1	0	0	0	1

Response:
    
        'summary' : 'Plants Bersimis-2, Brislay and Chute-Hemmings are currently facing anomalies in terms of power generation. Plant Beauharnois is facing erronous water flow rates.\n', 
        'thoughts': 'Four plants are currently generating energy way lower than its output. Plant Beauharnois has having higher that normal water flow rate and should be closely monitored. \n
                So, an inspection of the turbines or other equipment might be necessary if this continues to happen.'
    

====

Here is the current table information:
Table data:
    {table_data}

Response:
   
'''

NL_TO_PANDAS_QUERY_TEMPLATE = '''
You are a bot who specializes on converting nautral language to Pandas retrieval query.
Pandas is a python library for database management.
You are provided with a table schema with column names and their types. 
================
Table variable name: st.session_state.cumm_data_df
Table schema: 
    Unnamed: 0                            int64
    name                                 object
    type                                 object
    longitude                           float64
    latitude                            float64
    watershed                            object
    reservoir                            object
    timestamp                             int64
    datetime                             object
    operational_status                    int64
    capacity (mw)                         int64
    head (m)                            float64
    units                               float64
    water_flow_rate                     float64
    reservoir_level                     float64
    total_energy_output                 float64
    total_energy_output_predict_mean    float64
    total_energy_output_predict_std     float64
    co2_emissions                       float64
    anomaly_total_energy_output           int64
    anomaly_water_flow_rate               int64
    anomaly_co2_emissions                 int64
    anomaly_reservoir_level               int64
    anomaly                               int64
==================
Your job is to figure out a pandas query to fetch the data requested by the user.
Respond with a dictionary with the following keys : 'query'
Your response should not contain anything else.
Example 1:
    User input: How many times have Beaumont had anomaly in total energy production before timestamp 10?
Response:
    "query" : "st.session_state.cumm_data_df[(st.session_state.cumm_data_df['name']=='Beaumont') & (st.session_state.cumm_data_df[anomaly_total_energy_output]==1) & (st.session_state.cumm_data_df['timestamp']<10)]"

====

User input : {input}

Response : 
'''


RETRIEVE_REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
('system',"Given the above conversation history and the latest user input, \
 your task is to ONLY REWRITE the user input that can used as a standalone question. Respond with a json with \
 two keys 'original_input' and 'rephrased_input' " ),
MessagesPlaceholder(variable_name="chat_history"),
("user","{input}")
])


DOCUMENT_CHAIN_PROMPT = ChatPromptTemplate.from_messages([
("system", "Answer the user's questions based on the context below. \
 If you don't know the answer just say you dont know. Do not try to  \
 come up with something. \n \ Make sure that \
 your response can be supported by the information provided in the \
 context:\\n The context: \\n{context}"),
("user","{input}"),
])


RAG_PROMPT_TEMPLATE =  "Answer the user's questions based on the context provided.\n \
         If you don't know the answer just say you dont know. Do not try to come up with something. \n \
         Keep your answer brief and to the point. \n\n \
         The context: {context} \n \
         Human's question: {input}"


EMAIL_PROMPT_TEMPLATE= "Help the user write an email based on the user query provided below. \n \
        User query: {input} \n"