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
                    need to consult an electrical grid outage monitoring table or require assistance with creative writing. \n \
                    The table contains information regarding different locations within an electrical grid, \
                    the postal codes, locations, outage duration and details related to different areas of the region. \
                    \n Here is the input \n\n {input}.\n\n Repond with one word: 'outage', 'conv','qa', or 'writing'. \
                    'outage' if you think it would need to access the outage monitoring table, \
                    'conv', if you think it is a part of a regular conversation, \
                    'qa', if you think it would require fact based question answering capabilities, and \
                    'writing' if you understand that \
                    the input is requesting help with creative writing. \n \
                    Explain your answer.\n \
                    Respond with a json with two keys. 'response' and 'explaination'. 'response' should either be 'power', 'conv', 'qa' or 'writing'."

ROUTER_PROMPT_TEMPLATE_4 = "Given an input, decide whether responding to it would require fact based question answering capabilitesl, a part of a general conversation, \
                    need to consult an electrical grid outage monitoring table or require assistance with creative writing. \
                    Always check for compatibility in the outage category first. If you do not find a match, check for the rest. \n \
                    The outage monitoring table contains information regarding affected areas, \
                    the postal codes, outage duration, people affected,  start and end times of the outages. \
                    \n Here is the input \n\n {input}.\n\n Repond with one word: 'outage', 'conv','qa', or 'writing'. \
                    'outage' if you think it would need to access the outage monitoring table, \
                    'conv', if you think it is a part of a regular conversation, \
                    'qa', if you think it would require general question answering capabilities, and \
                    'writing' if you understand that the input is requesting help with creative writing. \n \
                    Explain your answer.\n \
                    Respond with a json with two keys. 'response' and 'explaination'. 'response' should either be 'outage', 'conv', 'qa' or 'writing'."

ROUTER_PROMPT_TEMPLATE_BASIC = "Given an input, decide whether responding to it would require fact based question answering capabilitesl, a part of a general conversation, \
                    need to consult an electrical grid outage monitoring table or require assistance with creative writing. \
                    Always check for compatibility in the outage category first. If you do not find a match, check for the rest. \n \
                    The outage monitoring table contains information regarding affected areas, \
                    the postal codes, outage duration, people affected,  start and end times of the outages. \
                    \n Here is the input \n\n {input}.\n\n Repond with one word: 'outage', 'conv', or 'writing'. \
                    'outage' if you think it would need to access the outage monitoring table, \
                    'conv', if you think it is a part of a regular conversation, \
                    'writing' if you understand that the input is requesting help with creative writing. \n \
                    Explain your answer.\n \
                    Respond with a JSON with two keys. 'response' and 'explaination'. 'response' should either be 'outage', 'conv', or 'writing'."


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
The table data you will read belongs to a power grid system. '1' under any column indicates the presence of the said entity, '0' indicates normal funtion. \n
Try to provide suggestions as to why the anomaly occured along with the summary. \n\n Just provide your thoughts. 
No need to ask for feedback. Always respond in third person.\\
Respond with a dictionary with two keys: 'summary' and 'thoughts'
====
Example 1:

Table data:
'            name  timestamp  anomaly_total_energy_output  anomaly_water_flow_rate  anomaly_co2_emissions  anomaly_reservoir_level  anomaly\n
122  Beauharnois        122                            0                        1                      0                        0        1\n
123  Bersimis-2         122                            1                        0                      0                        0        1\n
134  Brislay            122                            1                        0                      0                        0        1\n
135  Chute-Hemmings     122                            0                        0                      1                        0        1
Response:
    
        'summary' : 'Plants Bersimis-2 and Brislay are currently facing anomalies in terms of power generation. Plant Beauharnois is facing erronous water flow rates. Plant Chute-Hemmings is experiencing higher than usual CO2 emissions.\n', 
        'thoughts': 'Four plants are currently anomalies. Plant Beauharnois has having abnormal normal water flow rate and should be closely monitored. \n
                So, an inspection of the turbines or other equipment might be necessary if this continues to happen.'
    

====

Here is the current table information:
Table data:
    {table_data}

Response:
   
'''


TABLE_SUMMARIZER_TEMPLATE_OUTAGE = '''
You are a bot who specializes on reading and understanding information and answering user questions based on that. \n
The data you will receive will be of the following:
1. A table that holds data related to outages occured in a power grid system. \n
2. A number or a string.

Given a table, try to answer the user's query as best as you can from the information provided. Be truthful and do not make up facts. \n
Alternatively, if provided a number or a string, that is other relevant information or the exact answer itself to the user query. Use that to answer the user's question. \n
Just provide the response. No need to ask for feedback. Always respond in third person.\n

====
Example 1:
Information from DB:
    postal_code	city	timestamp	latitude	longitude	datetime	outage_category	people_affected	start_time	end_time    duration_in_minutes  
    J7V 9W3	PINCOURT	89	45.383112	-73.973367	2024-01-04 17:00:00	Equipment Failure	1799	2024-01-04 17:00:00	2024-01-04 18:00:00 60
    H7M 6C1	LAVAL	89	45.613052	-73.726199	2024-01-04 17:00:00	External Factors	3386	2024-01-04 17:00:00	2024-01-04 19:00:00 120
    G5R 6C7	RIVIERE-DU-LOUP	89	47.821264	-69.529921	2024-01-04 17:00:00	Equipment Failure	3224	2024-01-04 17:00:00	2024-01-04 19:00:00 120
    J6J 2R4	CHATEAUGUAY	89	45.359409	-73.722442	2024-01-04 17:00:00	External Factors	869	2024-01-04 17:00:00	2024-01-04 18:00:00 60
    G7B 3C6	LA BAIE	89	48.336874	-70.887938	2024-01-04 17:00:00	System Improvement	1724	2024-01-04 17:00:00	2024-01-04 18:00:00 60
User input: 
    Can you summarize the contents of the table. 
Response:
    On January 4, 2024, at 17:00, five power outages occurred across different cities in Quebec, impacting a total of 11,002 people. \
            The outages were caused by a combination of equipment failure, external factors, and system improvements. The most significant impact was in \
            Laval, where 3,386 people were affected for two hours due to external factors. Rivière-du-Loup also experienced a two-hour outage caused \
            by equipment failure, affecting 3,224 people. In Pincourt, an equipment failure left 1,799 people without power for an hour. \
            Châteauguay faced an external factor-related outage affecting 869 people for an hour, while in La Baie, a system improvement \
            outage impacted 1,724 people for an hour.

Example 2:
Information from DB:
    External Factors Equipment Failure Natural Cause
User input:
    What were the outage causes for the outages in Sherbrooke? 
Response:
    The outage causes for the outages in Montreal were External Factors, Equipment Failure and Natural Cause.

Example 3:
Information from DB:
    6
User input:
    How many outages in Saint Jerome lasted for over 4 hours? 
Response:
    Six outages in Saint Jerome lasted for over 4 hours.
====

Here is the current table information:
Information from DB:
    {info_from_db}
User input:
    {input}
Response:

'''


PRED_ASSISTANT_PROMPT_TEMPLATE = '''
You are a bot who is an expert at timeseries forecasting using Neural networks. The user would provide their requirements and \
you will have to respond with helpful suggestions to their questions. They might also ask you to evaluate the choices of traning\
parameters they have made. Keep the responses to the point and brief.

Past conversation : {history}
User parameter choices: {user_param_choices}
Current User input: {input}

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
    timestamp                             int64
    anomaly_total_energy_output           int64
    anomaly_water_flow_rate               int64
    anomaly_co2_emissions                 int64
    anomaly_reservoir_level               int64
    anomaly                               int64
==================
Your job is to figure out a pandas query to fetch the data requested by the user.
Your query should ALWAYS return a complete table.
Respond with a dictionary with the following keys : 'query'
Your response should not contain anything else.
Example 1:
    User query: How many times have Plant Beaumont had anomaly in total energy production before timestamp 10?
Response:
    "query" : "st.session_state.cumm_data_df[(st.session_state.cumm_data_df['name']=='Beaumont') & (st.session_state.cumm_data_df['anomaly_total_energy_output']==1) & (st.session_state.cumm_data_df['timestamp']<10)]"

Example 1:
    User query: "Describe the status of Plant Chelsea at timestamp 57?"
Response:
    "query" : "st.session_state.cumm_data_df[(st.session_state.cumm_data_df['name']=='Chelsea') & (st.session_state.cumm_data_df['timestamp']==57)]"

====

User query : {input}

Response : 
'''

NL_TO_PANDAS_QUERY_TEMPLATE_OUTAGE =  '''
You are a bot who specializes on converting nautral language to Pandas retrieval query.
Pandas is a python library for database management.
You are provided with a table schema with column names and their types. 
================
Table variable name: st.session_state.cur_data_df
Table schema: 
 #   Column               Non-Null Count  Dtype         
---  ------               --------------  -----         
 0   postal_code          18036 non-null  object        
 1   city                 18036 non-null  object        
 2   timestamp            18036 non-null  int64         
 3   latitude             18036 non-null  float64       
 4   longitude            18036 non-null  float64       
 5   datetime             18036 non-null  datetime64[ns]
 6   outage_reason        18036 non-null  object        
 7   people_affected      18036 non-null  int32         
 8   start_time           18036 non-null  datetime64[ns]
 9   end_time             18036 non-null  datetime64[ns]
 10  duration_in_minutes  18036 non-null  int32         
==================
Your job is to figure out a pandas query to fetch the data requested by the user. 
Respond with a dictionary with the following keys : 'query'
Your response should not contain anything else.
Example 1:
    User query: What was the last anomaly type that occured in the city of Montreal?
Response:
    "query" : "st.session_state.cur_data_df[(st.session_state.cur_data_df['city'].str.lower().str.contains('Montreal'.lower()) & (st.session_state.cur_data_df['timestamp']==st.session_state.cur_data_df['timestamp'].max())]"

Example 1:
    User query: "What areas are affected by Natural Cause starting from 2nd Jan 2024 at 4pm till 3rd Feb 2024 11am?
Response:
    "query" : "st.session_state.cur_data_df[(st.session_state.cur_data_df['outage_category']=='Natural Cause') & (pd.to_datetime(st.session_state.cur_data_df['start_time'])>=datetime.datetime(year=2024,month=1, day=2,hour=16)) & (pd.to_datetime(st.session_state.cur_data_df['start_time'])<=datetime.datetime(year=2024,month=2, day=3,hour=11))]"

====

User query : {input}

Response : 
'''

RETRIEVE_REPHRASE_PROMPT = '''
Given the above conversation history and the latest user input, \
 your task is to ONLY REWRITE the user input in the light of the historical context if necessary. \
 Include all necessary details. Keep the response short and to the point.\
 Always respond ONLY with a valid JSON containing \
 two keys 'original_input' and 'rephrased_input'. The response should be usable by json.loads() method.
==============================
Example:
 Past conversation:
    [HumanMessage(content='How many outages were reported in Montreal?'), 
     AIMessage(content='The number of reported outages in Montreal is seven.'), 
     HumanMessage(content='Out of these, which one lasted the longest?'), 
     AIMessage(content='The longest outage reported in Montreal lasted for 196 minutes.')]
 Current unser input: 'Can you provide a short report on that particular outage?' 
 Response:
    "{{
        'original_input': 'Can you provide a short report on that particular outage?',
        'rephrased_input': 'Can you provide a summary of the outage in Montreal that lasted the longest?'
    }}"
================================

 Past conversations : {chat_history}
 Current user input: {input}
'''

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