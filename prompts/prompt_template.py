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


PRED_ASSISTANT_PROMPT_TEMPLATE = "Answer the user's query absed"

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
You are a bot who specializes on reading tabular data and summarizing the contents. \n
The table data you will read holds data related to outages occured in a power grid system. \n
Summarize the contents of the rows emphasizing on 'people_affected', 'outage_category' and their respective 'start_time'. \n  

\n\n Just provide your thoughts. No need to ask for feedback. Always respond in third person.\n
Respond ONLY with a dictionary with two keys: 'summary' and 'thoughts'.
====
Example 1:

    Table data:
        postal_code	timestamp	latitude	longitude	datetime	outage_category	people_affected	start_time	end_time
        G5Y 6N1	89	46.137345	-70.678584	2024-01-04 17:00	System Improvement	3194	2024-01-04 17:00	2024-01-04 19:00
        H9B 1T2	89	45.500104	-73.790448	2024-01-04 17:00	System Improvement	386	2024-01-04 17:00	2024-01-04 19:00
        J2E 1C7	89	45.905074	-72.534624	2024-01-04 17:00	Natural Cause	314	2024-01-04 17:00	2024-01-04 19:00
        H7C 1N3	89	45.611089	-73.645161	2024-01-04 17:00	Environmental Factors	4910	2024-01-04 17:00	2024-01-04 19:00
        G3G 2Y8	89	46.894125	-71.373364	2024-01-04 17:00	Power System Repair	1442	2024-01-04 17:00	2024-01-04 19:00
        H9H 4A8	89	45.469722	-73.856587	2024-01-04 17:00	System Improvement	455	2024-01-04 17:00	2024-01-04 19:00
        H9J 1L7	89	45.453199	-73.864731	2024-01-04 17:00	External Factors	4238	2024-01-04 17:00	2024-01-04 19:00

    Response:
        "summary" : "On January 4, 2024, multiple outages were reported across various locations, affecting a total of 18,939 people. \
            The outages were categorized under system improvement, natural causes, environmental factors, power system repair, and external factors. \
                Each outage began at 17:00 and was resolved by 19:00 on the same day. The largest outage affected 4,910 people in the postal code H7C 1N3, \
                    attributed to environmental factors.", 
        "thoughts": "The simultaneous occurrence of these outages across different regions suggests a coordinated effort, likely aimed at modernizing and \
                reinforcing the power grid. While the planned outages for system improvements are a positive sign of proactive maintenance, the disruptions \
                    caused by natural and environmental factors reveal underlying vulnerabilities. It's clear that while the grid is evolving, nature remains a formidable challenge, \
                reminding us that our infrastructure must not only be advanced but also resilient. The large-scale impact in areas affected by external and environmental \
                    factors indicates that these regions may benefit from more aggressive investments in grid hardening, such as weatherproofing and strategic \
                    vegetation management. In the long term, integrating predictive analytics and real-time monitoring could reduce the frequency and impact of such outages, \
                            enhancing overall grid stability and customer satisfaction. The brief, yet impactful, two-hour window of these outages also suggests a well-organized \
                                response team, capable of restoring power swiftly—an encouraging sign for future incidents."    

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
    timestamp                             int64
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
Table variable name: st.session_state.cumm_data_df
Table schema: 
    postal_code         object
    timestamp            int64
    latitude           float64
    longitude          float64
    datetime            object
    outage_category     object
    people_affected      int64
    start_time          object
    end_time            object
==================
Your job is to figure out a pandas query to fetch the data requested by the user.
Respond with a dictionary with the following keys : 'query'
Your response should not contain anything else.
Example 1:
    User query: What was the last anomaly type that occured at H9B 1T2?
Response:
    "query" : "st.session_state.cumm_data_df[(st.session_state.cumm_data_df['postal_code']=='H9B 1T2') & (st.session_state.cumm_data_df['timestamp']==st.session_state.cumm_data_df['timestamp'].max())]"

Example 1:
    User query: "What areas are affected by Natural Cause starting from 2nd Jan 2024 at 4pm till 3rd Feb 2024 11am?
Response:
    "query" : "st.session_state.cumm_data_df[(st.session_state.cumm_data_df['outage_category']=='Natural Cause') & (pd.to_datetime(st.session_state.cumm_data_df['start_time'])>=datetime.datetime(year=2024,month=1, day=2,hour=16)) & (pd.to_datetime(st.session_state.cumm_data_df['start_time'])<=datetime.datetime(year=2024,month=2, day=3,hour=11))]"

====

User query : {input}

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