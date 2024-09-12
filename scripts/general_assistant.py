
import os, tempfile
from pathlib import Path
import streamlit as st
import pytesseract

from langchain.vectorstores import Chroma

#import for llms
from langchain_community.llms import Ollama

#import for embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma 

from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from prompts.prompt_template import *
from utils.utils import *
import ipdb 
import glob
import warnings
warnings.filterwarnings("ignore")

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import torch
import numpy as np
import spacy 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time 
from streamlit_extras.stylable_container import stylable_container
from st_aggrid import AgGrid
import pandas as pd
import re 
from langchain_openai import ChatOpenAI

nlp = spacy.load("en_core_web_sm")

VECTOR_DB_PATHS = {
                'Public' : Path('../vectorstores/energy_public'), 
                'Private' : Path('../vectorstores/energy_private'),
                    }

# LOCAL_VECTOR_STORE_DIR = Path('../vectorstore')

# PERSIST_DIRECTORY = Path('../vectorstore_test')
TMP_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

##############################  backend functions  ###############################

#for the vector store


# def load_documents():
#     document_list = []
#     doc_files = glob.glob(f'{TMP_DIR.as_posix()}/*.pdf')
#     for doc in doc_files:
#         document_list.extend(PyPDFLoader(doc).load())
#     return document_list


# def split_documents(documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
#     texts = text_splitter.split_documents(documents)
#     return texts

# def create_vector_db(texts):
#     vectordb = Chroma.from_documents(texts, embedding=HuggingFaceEmbeddings())
#     vectordb = Chroma(persist_directory=VECTOR_DB_PATHS['Public'].as_posix(), embedding_function=HuggingFaceEmbeddings())
#     #vectordb.persist()
#     return vectordb

# def update_vector_db():
#     st.session_state.vector_db = Chroma(persist_directory=VECTOR_DB_PATHS['Public'].as_posix(), embedding_function=HuggingFaceEmbeddings())
 

def setup_llms_assistant():

    st.session_state.llm_model_chat = Ollama(model='llama3.1', system='You are a helpful question answering bot.')
    st.session_state.llm_model_instruct = Ollama(model='llama3.1', temperature=0.1, format='json', system="You are an LLM who is logical and is excellent at following instructions.")
    with open('../assets/openai_api_key.txt', 'r') as f:
        key = f.read()
    os.environ["OPENAI_API_KEY"]=key
    st.session_state.llm_openai = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_retries=2,
        # api_key="...",
        # base_url="...",
        # organization="...",
        # other params...
    )
    
    # st.session_state.llm_dashboard_assistant = Ollama(model='llama3.1', format='json', system="You are a bot who specializes on reading tabular data, summarizing them and providing insights.")
    st.session_state.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") #chroma default embedding model


def setup_llm_chains_assistant():

    #build the conversation chain
    conv_prompt = PromptTemplate(input_variables=['input', 'history'], template=CONV_PROMPT_TEMPLATE)
    st.session_state.conv_chain = LLMChain(llm=st.session_state.llm_model_chat, prompt=conv_prompt, output_key='answer')

    #build the rephrase chain 
    rephrase_prompt = PromptTemplate(input_variables=['input', 'chat_history'], template=RETRIEVE_REPHRASE_PROMPT_GA)

    st.session_state.rephrase_chain = LLMChain(llm=st.session_state.llm_openai, prompt=rephrase_prompt)


    #build the document chain
    st.session_state.document_chain=create_stuff_documents_chain(st.session_state.llm_model_chat, prompt = DOCUMENT_CHAIN_PROMPT)

    #build the router chain
    router_prompt = PromptTemplate(
        input_variables=["input"], template=ROUTER_PROMPT_TEMPLATE_2
    )
    st.session_state.router_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=router_prompt, output_key='answer')

    #setup the email writing chain
    email_prompt = PromptTemplate(input_variables=['input'], template=EMAIL_PROMPT_TEMPLATE)
    st.session_state.email_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=email_prompt, output_key='answer')    

def load_vectordbs():
    st.session_state.private_db = Chroma(persist_directory=VECTOR_DB_PATHS['Private'].as_posix(), 
                                         embedding_function=HuggingFaceEmbeddings())
    
    st.session_state.private_db_docs = set([elem['source'] for elem in st.session_state.private_db.get(include=['metadatas'])['metadatas']])

    st.session_state.public_db = Chroma(persist_directory=VECTOR_DB_PATHS['Public'].as_posix(), 
                                        embedding_function=HuggingFaceEmbeddings())
    
    st.session_state.public_db_docs = set([elem['source'] for elem in st.session_state.public_db.get(include=['metadatas'])['metadatas']])


def get_db_files(db):
    filenames = list(set([elem['source'] for elem in db.get(include=['metadatas'])['metadatas']]))
    fnames_only = [doc_name.split('\\')[-1] for doc_name in filenames]
    return pd.DataFrame({'filename': fnames_only})


def get_relevant_documents_from_dbs(query_text):
    rel_docs_and_score_pvt = st.session_state.private_db.similarity_search_with_score(query_text, 
                                                                        k=st.session_state.search_k,
                                                                        )
    rel_docs_and_score_pub = st.session_state.public_db.similarity_search_with_score(query_text, 
                                                                    k=st.session_state.search_k,
                                                                    )
    docs_and_scores = rel_docs_and_score_pub + rel_docs_and_score_pvt


    docs_and_scores.sort(key=lambda x:x[1])
    return [item[0] for item in docs_and_scores[0:st.session_state.search_k]]


def check_sentence_hallucination(query, context, response, sample_size=5):
    '''
    Given the query and context used to generate the original respose
    and the response, check for hallucination on a sentence level
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selfcheck_nli = SelfCheckNLI(device=device) # set device to 'cuda' if GPU is available

    sample_responses = []
    #generate sample answers
    for i in range(sample_size):
        samp_resp = st.session_state.document_chain.invoke({'input':query, 
                                                'context':context})
        sample_responses.append(samp_resp)

    #break into sentence
    resp_sentences = [sent.text.strip() for sent in nlp(response).sents] # spacy sentence tokenization

    sent_scores_nli = selfcheck_nli.predict(
        sentences = resp_sentences,                          # list of sentences
        sampled_passages = sample_responses, # list of sampled passages
    )
    return resp_sentences, sent_scores_nli


def check_sentence_hallucination_cosine_similarity(
                                                    context, 
                                                    response, 
                                                   ):
  
    context_page_content = [doc.page_content for doc in context]
    resp_sentences = [sent.text.strip() for sent in nlp(response).sents] # spacy sentence tokenization
    sent_embeddings = st.session_state.embedding_model.encode(resp_sentences)
    context_embeddings = st.session_state.embedding_model.encode(context_page_content)
    sentence_cosine_scores = cosine_similarity(sent_embeddings, context_embeddings)
    return resp_sentences, np.max(sentence_cosine_scores, axis=1)


def annotate_response(reponse_sentences, scores, hallu_method='selfcheckgpt'):
    anno_result = ""

    if hallu_method=='selfcheckgpt':
        for sent, score in zip(reponse_sentences, scores):
            if score >= 0.6: #0 is no hallu, 1 is hallu / for cosine sim: 0 is hallu, 1 is
                sent = f":red-background[{sent}]"
            if score > 0.35 and score < 0.6:
                sent = f":orange-background[{sent}]"

            anno_result += sent  
    if hallu_method=='cosine_similarity':
        for sent, score in zip(reponse_sentences, scores):
            if score < 0.5: #0 is no hallu, 1 is hallu / for cosine sim: 0 is hallu, 1 is not
                sent = f":red-background[{sent}]"
            if score > 0.5 and score < 0.65:
                sent = f":orange-background[{sent}]"

            anno_result += sent 
    return anno_result


def query_chain_general_assistant():
    #run the email chain

    query_text = st.session_state.current_input

    #use chains

    #check if retrieval is required
    input_dict = {'input': query_text, 'chat_history': get_session_gen_assist_chat_history()}

    resp = st.session_state.rephrase_chain.invoke(input_dict)
    resp_string = get_key_val_from_llm_json_string(resp['text'], 'rephrased_input')
    print("\n\n\n**********Rephrased input :", resp_string)
    print("\n\n\n ****** CHAT HISTORY :", get_session_gen_assist_chat_history())


    resp = st.session_state.router_chain.invoke({'input': resp_string})
    print("***RESPONSE QA : ", resp['answer'])
    is_qa = get_key_val_from_llm_json_string(resp['answer'], 'response')
    
    
    if is_qa.strip().lower()=='qa':

        #rephrase question using history
       
        #use response to retrieve relevant documents 
        docs = []
        if st.session_state.use_kb:
            docs = get_relevant_documents_from_dbs(resp_string)
            print("\n\n\n************ Docs in the context :", docs)
        #get answer using relevant documents and question
        result = st.session_state.document_chain.invoke({'input':resp_string, 
                                                'context':docs})
        

        #annotate the response with hallucination information
        if st.session_state.use_kb:
            regex = re.compile("[^a-zA-Z0-9.,!' $\n\-():]")
            result_clean = regex.sub('', result)
            start_time = time.time()
            #resp_sent, scores = check_sentence_hallucination(resp_string, docs, result, sample_size=5)
            if st.session_state.use_hallu_detect:
                resp_sent, scores = check_sentence_hallucination_cosine_similarity(docs, result_clean)
                print("\n\n\n Execution time : ", time.time()-start_time)
                anno_result = annotate_response(resp_sent, scores, 'cosine_similarity')
                # print("Scores ***************", scores)
                # print("RESULT ***************", result)
                # print("RESULT ***************", anno_result)
            else:
                anno_result = result
        else:
            anno_result = result
        #result = anno_result
   
        #anno_result = result
        st.session_state.response = anno_result
        st.session_state.response_context = docs

    if is_qa.strip().lower()=='conv':
        result = st.session_state.conv_chain.invoke(input_dict)
        anno_result = result['answer']
        st.session_state.response = result
        st.session_state.response_context = ""    
    

    if is_qa.strip().lower()=='writing':
        resp = st.session_state.rephrase_chain.invoke(input_dict)
        resp_string = get_key_val_from_llm_json_string(resp['text'], 'rephrased_input')
        print("**********Rephrased input :", resp_string)
        result = st.session_state.email_chain.invoke({'input':resp_string})
        anno_result = result['answer']
        st.session_state.response = result
        st.session_state.response_context = ""    
    
    #save the query in the chat history
    st.session_state.messages_gen_assist.append({"speaker" : "user", "content": query_text})
    
    # rel_sources = [doc.metadata['source'] for doc in docs]
    # rel_pages = [doc.metadata['page'] for doc in docs]
    # rel_data_resp = f'\n Relevant information can be found in the following documents : {" ".join(rel_sources)}'
    st.session_state.messages_gen_assist.append({"speaker" : "AI",
                                    "content": anno_result})
    

################################  front end functions  ################################
def get_session_gen_assist_chat_history():
    chat_list = st.session_state.messages_gen_assist 
    chat_history = []
    for conv in chat_list:
        if conv['speaker']=="user":
            chat_history.append(HumanMessage(content=conv['content']))
        if conv['speaker']=='AI':
            chat_history.append(AIMessage(content=conv['content']))
    return chat_history

@st.experimental_fragment
def build_chatbot_params_console():
    with stylable_container(key='bot_param_header',
                            css_styles='''
                            {
                                background-color: #ddd7d7;
                                padding: 0;
                            }
                            '''):
        st.markdown("<h3 style='font-family: sans-serif; text-align: center; color: black;'> Bot parameter console</h3>", unsafe_allow_html=True)
    st.session_state.use_kb = st.toggle("Use Knowledge base")
    st.session_state.use_hallu_detect = st.toggle("Check for hallucination")
    k_list = [3,4,5,6,7]
    st.session_state.search_k = st.selectbox('No. of documents in context:', k_list)


def load_documents(filepaths):
    document_list = []
    for doc in filepaths:
        if doc.split('.')[-1]=='pdf':
            document_list.extend(PyPDFLoader(doc).load())
        elif doc.split('.')[-1]=='txt':
            document_list.extend(TextLoader(doc).load())
    return document_list

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    return texts

@st.experimental_fragment
def process_documents():
    if not st.session_state.uploaded_files:
        st.warning(f"Please upload the documents first.")
    else:
        for source_doc in st.session_state.uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), 
                                             prefix=source_doc.name.split('.')[0],
                                             suffix='.pdf') as tmp_file:
                tmp_file.write(source_doc.read())
        
        temp_files = glob.glob(f'{TMP_DIR}/*')
        with st.spinner("Loading documents . . ."):
            documents = load_documents(temp_files)
        for _file in TMP_DIR.iterdir():
            temp_file = TMP_DIR.joinpath(_file)
            temp_file.unlink()
        
        with st.spinner("Parsing text . . ."):
            texts = split_documents(documents)
        st.session_state.uploaded_files = []
        st.session_state.private_db.add_documents(texts)

@st.experimental_fragment
def delete_files_from_db():
    rel_ids = []
    private_db_metadata = st.session_state.private_db._collection.get(include=['metadatas'])

    for fname in st.session_state.db_del_files:
        rel_ids_file = []
        for id_val, doc_metadata in zip(private_db_metadata['ids'], private_db_metadata['metadatas']):
            if fname==doc_metadata['source'].split('\\')[-1]:
                rel_ids_file.append(id_val)
        rel_ids.extend(rel_ids_file)
    if len(rel_ids):
        st.session_state.private_db._collection.delete(rel_ids)
    st.session_state.db_del_files = []


@st.experimental_fragment
def build_doc_management_console():
    with stylable_container(key='doc_management_header',
                            css_styles='''
                            {
                               background-color: #ddd7d7;
                               padding: 0;
                            }
                            '''):
        st.markdown("<h3 style='font-family: sans-serif; text-align: center; color: black;'> Doc management</h3>", unsafe_allow_html=True)
    st.session_state.uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
    st.button("Submit documents", on_click=process_documents)
    st.session_state.db_view_selected = st.selectbox("View files in Database:", ['Public', 'Private'])
    with st.popover("View files"):
        if st.session_state.db_view_selected=='Public':
            files_db = get_db_files(st.session_state.public_db)
        else:
            files_db = get_db_files(st.session_state.private_db)
        AgGrid(files_db)

        if st.session_state.db_view_selected=='Public':
            st.button("Cannot delete public files", on_click=delete_files_from_db, disabled=True)
        else:
            st.session_state.db_del_files = st.multiselect('Select :',options=files_db, placeholder='Choose files')
            st.button("Delete ", on_click=delete_files_from_db)

@st.experimental_fragment
def build_context_display_window():
    row_container_list = []
    if 'response_context' in st.session_state and st.session_state.response_context !="":

        for _ in st.session_state.response_context:
            row_container_list.append(stylable_container(key='context_data',
                            css_styles='''
                            {
                               background-color:  #e1e1ea;
                               padding: 5px;
                               height: 65px;
                               border-radius: 10px;
                            }
                            '''))
    
    i = 0
    if 'response_context' in st.session_state and st.session_state.response_context !="":
        for context_doc in st.session_state.response_context:
            context_metadata = context_doc.metadata
            with row_container_list[i]:
                context_disp_col, context_disp_col2 = st.columns([0.8, 0.2])

                with context_disp_col:
                    data_source = context_metadata['source'].split('\\')[-1]
                    st.markdown(f"<p style='font-size: 1.2em'> <b>Source: </b> {data_source} &nbsp &nbsp &nbsp <b>Page: </b> {context_metadata['page']}</p>",  unsafe_allow_html=True)
                with context_disp_col2:
                    with st.popover("View page contents"):
                        st.write(context_doc.page_content)

            i+=1

def main():
    # page title
    st.set_page_config(
        page_title="EnergyGPT Dashboard",
        page_icon="✅",
        layout="wide",
    )
    st.title(
        'Helper bot'
    )
    setup_llms_assistant()
    setup_llm_chains_assistant()
    load_vectordbs()    
# App logic
    #uploaded_file = st.session_state.source_docs


    if "messages_gen_assist" not in st.session_state:
        st.session_state.messages_gen_assist = []

    col1, col2 = st.columns([0.3,0.7], gap="small")
    with col1:
        respose_gen_console =  st.container(height=280)  
        document_management_console =  st.container(height=620)
    with col2:
        chat_window =  st.container(height=500)  
        context_display_console = st.container(height=400)
    with chat_window:
        st.chat_input(placeholder = 'Ask me anything: From writing emails to finding answers from documents. ', 
                        on_submit=query_chain_general_assistant,
                        key='current_input')

        #display the chat history so far
        with st.container(height=400):
            for msg in st.session_state.messages_gen_assist:
                st.chat_message(msg['speaker']).markdown(msg['content'])

    #display the documents in the context used to come up with the answer
    with context_display_console:
        build_context_display_window()
    with respose_gen_console:
        build_chatbot_params_console()

    with document_management_console:
        build_doc_management_console()

if __name__=='__main__':
  main()