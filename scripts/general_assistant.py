
import os, tempfile
from pathlib import Path
import streamlit as st
import pytesseract

from langchain.vectorstores import Chroma

#import for llms
import ollama
from langchain_community.llms import Ollama
from utils.llm_apis import LocalOllama, OllamaChain
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
from utils.hallucination_detection import CosineDetector, DeepEvalDetector
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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

nlp = spacy.load("en_core_web_sm")


VECTOR_DB_PATHS = Path('../vectorstores/finance_QA_openaiembedding')


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
 
@st.cache_resource
def setup_llms_assistant():

    st.session_state.llm_model_chat = LocalOllama(model='llama3.2', system='You are a helpful question answering bot.')
    st.session_state.llm_model_instruct = LocalOllama(model='llama3.2', temperature=0.1, format='json', system="You are an LLM who is logical and is excellent at following instructions.")
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

@st.cache_resource
def setup_llm_chains_assistant():

    #build the conversation chain
    conv_prompt = PromptTemplate(input_variables=['input', 'history'], template=CONV_PROMPT_TEMPLATE)
    #st.session_state.conv_chain = OllamaChain(llm=st.session_state.llm_model_chat, prompt=conv_prompt)
    st.session_state.conv_chain_ga = LLMChain(llm=st.session_state.llm_openai, prompt=conv_prompt)

    #build the rephrase chain 
    rephrase_prompt = PromptTemplate(input_variables=['input', 'chat_history'], template=RETRIEVE_REPHRASE_PROMPT_GA)
    st.session_state.rephrase_chain = LLMChain(llm=st.session_state.llm_openai, prompt=rephrase_prompt)

    #build the document chain
    #st.session_state.rag_prompt = DOCUMENT_CHAIN_PROMPT
    st.session_state.document_chain=LLMChain(llm=st.session_state.llm_openai,  prompt = DOCUMENT_CHAIN_PROMPT)
    sample_q_prompt = PromptTemplate(input_variables=['input', 'chat_history'], template=SAMPLE_QUESTION_GENERATION_PROMPT_GA)

    st.session_state.question_sampler_chain=LLMChain(llm=st.session_state.llm_openai,  prompt = sample_q_prompt)

    #build the router chain
    router_prompt = PromptTemplate(
        input_variables=["input"], template=ROUTER_PROMPT_TEMPLATE_2
    )
    st.session_state.router_chain_assistant = LLMChain(llm=st.session_state.llm_openai, prompt=router_prompt)

    #setup the email writing chain
    email_prompt = PromptTemplate(input_variables=['input'], template=EMAIL_PROMPT_TEMPLATE)
    st.session_state.email_chain = LLMChain(llm=st.session_state.llm_openai, prompt=email_prompt)    


@st.cache_resource
def load_vectordbs():
    st.session_state.finance_db = Chroma(persist_directory=VECTOR_DB_PATHS.as_posix(), 
                                         embedding_function=OpenAIEmbeddings())
    

@st.cache_resource 
def load_hallucination_detector():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    #st.session_state.hallucination_detector = CosineDetector(embedding_model)
    st.session_state.hallucination_detector = DeepEvalDetector('faithfulness',
                                                               'gpt-4o-mini' )


def get_db_files(db):
    filenames = list(set([elem['source'] for elem in db.get(include=['metadatas'])['metadatas']]))
    fnames_only = [doc_name.split('\\')[-1] for doc_name in filenames]
    return pd.DataFrame({'filename': fnames_only})


def get_relevant_documents_from_dbs(query_text):
    rel_docs_and_score = st.session_state.finance_db.similarity_search_with_score(query_text, 
                                                                        k=st.session_state.search_k,
                                                                        )
    return [tup[0] for tup in rel_docs_and_score]


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


def annotate_response(response_sentences, scores, hallu_method='deepeval'):
    anno_result = ""
    if hallu_method in ['cosine_similarity', 'deepeval']:
        for sent, score in zip(response_sentences, scores):
            if score <= 0.25: #0 is no hallu, 1 is hallu / for cosine sim: 0 is hallu, 1 is not
                sent = f"""
                        <div class="hover-text highlight-red">
                            {sent}
                            <div class="hover-message">"Severe Error"</div>
                        </div>
                        """
            if score > 0.25 and score <= 1:
                sent = f"""
                        <div class="hover-text highlight-violet">
                            {sent}
                            <div class="hover-message">"Mild Error"</div>
                        </div>
                        """
            anno_result += sent 

    return anno_result


def query_chain_general_assistant():
    #run the email chain

    query_text = st.session_state.current_input

    #use chains

    #check if retrieval is required
    chat_history = get_session_gen_assist_chat_history()
    rephrase_resp = st.session_state.rephrase_chain.invoke({'input': query_text, 'chat_history': chat_history})
    print("****************Rephrase response*********************", rephrase_resp['text'])

    resp_string = get_key_val_from_llm_json_string(rephrase_resp['text'], 'rephrased_input')

    router_resp = st.session_state.router_chain_assistant.invoke({'input': resp_string})


    print("****************Router response*********************", router_resp['text'])

    print("***RESPONSE QA : ", router_resp['text'])
    is_qa = get_key_val_from_llm_json_string(router_resp['text'], 'response')
    
    
    if is_qa.strip().lower()=='qa':

        #rephrase question using history
       
        #use response to retrieve relevant documents 
        docs = []
        if st.session_state.use_kb:
            docs = get_relevant_documents_from_dbs(resp_string)
            print("\n\n\n************ Docs in the context :", docs)
        #get answer using relevant documents and question        
        rag_response = st.session_state.document_chain.invoke({'input':resp_string, 
                                                'context':docs})['text']
        
        sample_questions=''
        if st.session_state.use_kb and st.session_state.generate_sample_questions:
            sample_questions = st.session_state.question_sampler_chain.invoke({'input': resp_string, 'context': docs})['text']
            sample_questions = '\n Other related questions include: \n'+sample_questions
        
        print("****************RAG response*********************", rag_response)
        print("*****************Sample questions***************", sample_questions)

        #annotate the response with hallucination information
        if st.session_state.use_kb:
            regex = re.compile("[^a-zA-Z0-9.,!' $\n\-():]")
            result_clean = regex.sub('', rag_response)
            start_time = time.time()
            
            #resp_sent, scores = check_sentence_hallucination(resp_string, docs, result, sample_size=5)
            if st.session_state.use_hallu_detect:
                resp_sent, scores = st.session_state.hallucination_detector.check_hallucination(resp_string, result_clean, docs)
                print("\n\n\n Execution time : ", time.time()-start_time)
                anno_result = annotate_response(resp_sent, scores, 'deepeval')
                for s, score in zip(resp_sent, scores):
                    print(f"*****************Sentence: {s} \n Scores ***************: {score}")
                # print("RESULT ***************", result)
                # print("RESULT ***************", anno_result)
            else:
                anno_result = rag_response
        else:
            anno_result = rag_response
        #result = anno_result
   
        #anno_result = result
        anno_result = anno_result+'\n'+sample_questions
        st.session_state.response = anno_result
        st.session_state.response_context = docs

    if is_qa.strip().lower()=='conv':
        conv_resp = st.session_state.conv_chain_ga.invoke({'input': query_text, 'chat_history': chat_history})['text']
        print("****************Conv response*********************", conv_resp)

        anno_result = conv_resp
        st.session_state.response = conv_resp
        st.session_state.response_context = ""    
    

    if is_qa.strip().lower()=='writing':

        email_resp = st.session_state.email_chain.invoke({'input':resp_string})['text']
        print("****************Conv response*********************", email_resp)

        anno_result = email_resp
        st.session_state.response = email_resp
        st.session_state.response_context = ""    
    
    #save the query in the chat history
    st.session_state.messages_gen_assist.append({"speaker" : "user", "content": query_text})
    
    # rel_sources = [doc.metadata['source'] for doc in docs]
    # rel_pages = [doc.metadata['page'] for doc in docs]
    # rel_data_resp = f'\n Relevant information can be found in the following documents : {" ".join(rel_sources)}'
    st.session_state.messages_gen_assist.append({"speaker" : "AI",
                                    "content": re.sub('\$','\\$',anno_result)
})
    

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

@st.fragment
def build_chatbot_params_console():
    with stylable_container(key='bot_param_header',
                            css_styles='''
                            {
                                background-color: white;
                                padding: 0;
                            }
                            '''):
        st.markdown("<h3 style='font-family: sans-serif; text-align: center; color: black;'> Generation Parameters</h3>", unsafe_allow_html=True)
    st.session_state.use_kb = st.toggle("Use Knowledge base")
    st.session_state.use_hallu_detect = st.toggle("Check for hallucination")
    st.session_state.show_supporting_docs = st.toggle("Show supporting documents")
    st.session_state.generate_sample_questions = st.toggle("Suggest related questions")
    k_list = [5,6,7,8,9]
    st.session_state.search_k = st.selectbox('No. of documents in context:', k_list, index = len(k_list)-1)


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


@st.fragment
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
        st.success("File(s) added successfully!")

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
    st.success("File(s) deleted successfully!")
    st.rerun(scope="fragment")

def popover_logic():
    if st.session_state.db_view_selected=='Public':
        files_db = get_db_files(st.session_state.public_db)
    else:
        files_db = get_db_files(st.session_state.private_db)
    AgGrid(files_db)

    if st.session_state.db_view_selected=='Public':
        pass
    else:
        st.session_state.db_del_files = st.multiselect('Select :',options=files_db, placeholder='Choose files')
        delete_file_button = st.button("Delete ")
        if delete_file_button:
            delete_files_from_db()


@st.fragment
def build_doc_management_console():
    with stylable_container(key='doc_management_header',
                            css_styles='''
                            {
                               background-color: white;
                               padding: 0;
                            }
                            '''):
        st.markdown("<h3 style='font-family: sans-serif; text-align: center; color: black;'> Document Management</h3>", unsafe_allow_html=True)
    st.session_state.uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
    add_doc_button = st.button("Submit documents")
    if add_doc_button:
        process_documents()
    st.session_state.db_view_selected = st.selectbox("View files in Database:", ['Public', 'Private'])

    view_files =  st.popover("View files")
    with view_files:
        popover_logic()
        # if st.session_state.db_view_selected=='Public':
        #     files_db = get_db_files(st.session_state.public_db)
        # else:
        #     files_db = get_db_files(st.session_state.private_db)
        # AgGrid(files_db)

        # if st.session_state.db_view_selected=='Public':
        #     pass
        # else:
        #     st.session_state.db_del_files = st.multiselect('Select :',options=files_db, placeholder='Choose files')
        #     delete_file_button = st.button("Delete ")
        #     if delete_file_button:
        #         delete_files_from_db()

@st.fragment
def build_context_display_window():
    row_container_list = []
    if 'response_context' in st.session_state and st.session_state.response_context !="":
        st.markdown("<h3 style='font-family: sans-serif; text-align: left; color: black;'> Supporting documents :</h3>", unsafe_allow_html=True)

        for _ in st.session_state.response_context:
            row_container_list.append(stylable_container(key='context_data',
                            css_styles='''
                            {
                               background-color:  #f4f7ff;
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
                context_disp_col2 = st.container()

                with context_disp_col2:
                    with st.popover("View page contents"):
                        st.write(context_doc.page_content)

            i+=1

def build_main_page_general_assistant():

    setup_llms_assistant()
    setup_llm_chains_assistant()
    load_vectordbs()    
    load_hallucination_detector()
    st.markdown("""
        <style>
        .highlight-violet {
            background-color: violet;
            color: black;
            padding: 1px 2px;
            border-radius: 3px;
        }
        .highlight-red {
            background-color: red;
            color: black;
            padding: 1px 2px;
            border-radius: 3px;
        }
        .hover-text {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        .hover-text .hover-message {
            visibility: hidden;
            width: 100px;
            background-color: black;
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 100%;
            left: 50%;
            margin-left: -100px;
            opacity: 0.2;
            transition: opacity 0.3s;
        }
        .hover-text:hover .hover-message {
            visibility: visible;
            opacity: 0.8;
        }
        </style>
    """, unsafe_allow_html=True)
# App logic
    #uploaded_file = st.session_state.source_docs


    if "messages_gen_assist" not in st.session_state:
        st.session_state.messages_gen_assist = []

    col1, col2 = st.columns([0.3,0.7], gap="small")
    with col1:
        respose_gen_console =  st.container(height=320, border=False)  
        document_management_console =  st.container(height=580,  border=False)
    
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
                    with st.chat_message(msg['speaker']):
                        st.markdown(msg['content'], unsafe_allow_html=True)
        else:
            with st.container(height=800):
                for msg in st.session_state.messages_gen_assist:
                    with st.chat_message(msg['speaker']):
                        st.markdown(msg['content'], unsafe_allow_html=True)
    #display the documents in the context used to come up with the answer\
    
    if st.session_state.show_supporting_docs:
        with context_display_console:
                build_context_display_window()
    # with document_management_console:
    #     build_doc_management_console()


def main():
    # page title
    st.set_page_config(
        page_title="EnergyGPT Dashboard",
        page_icon="✅",
        layout="wide",
    )
    st.title(
        'General Assistance'
    )
    st.html("../css/dashboard_styles.html")
    build_main_page_general_assistant()
    

if __name__=='__main__':
  main()