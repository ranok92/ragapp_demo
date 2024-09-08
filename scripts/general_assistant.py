
import os, tempfile
from pathlib import Path
import streamlit as st
import pytesseract

from langchain.vectorstores import Chroma

#import for llms
from langchain_community.llms import Ollama

#import for embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain.document_loaders import PyPDFLoader
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
 
@st.cache_resource
def combine_vector_dbs(path1, path2):
    vectordb1 = Chroma(persist_directory=path1.as_posix(), embedding_function=HuggingFaceEmbeddings())
    vectordb2 = Chroma(persist_directory=path2.as_posix(), embedding_function=HuggingFaceEmbeddings())
    vector_db2_data = vectordb2._collection.get(include=['documents', 'metadatas', 'embeddings'])
    vectordb1._collection.add(
        embeddings=vector_db2_data['embeddings'],
        metadatas=vector_db2_data['metadatas'],
        documents=vector_db2_data['documents'],
        ids=vector_db2_data['ids']
    )
    return vectordb1


def setup_llms_assistant():

    st.session_state.llm_model_chat = Ollama(model='llama3.1',  temperature = 0.2, system='You are a helpful question answering bot.')
    st.session_state.llm_model_instruct = Ollama(model='llama3.1', temperature = 0.2, format='json', system="You are an LLM who is logical and is excellent at following instructions.")
    # st.session_state.llm_dashboard_assistant = Ollama(model='llama3.1', format='json', system="You are a bot who specializes on reading tabular data, summarizing them and providing insights.")
    st.session_state.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") #chroma default embedding model


def setup_llm_chains_assistant():

    #build the conversation chain
    conv_prompt = PromptTemplate(input_variables=['input', 'history'], template=CONV_PROMPT_TEMPLATE)
    st.session_state.conv_chain = LLMChain(llm=st.session_state.llm_model_chat, prompt=conv_prompt, output_key='answer')

    #build the rephrase chain 
    rephrase_prompt = PromptTemplate(input_variables=['input', 'chat_history'], template=RETRIEVE_REPHRASE_PROMPT_GA)

    st.session_state.rephrase_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=rephrase_prompt)


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

def query_chain():
    #run the email chain

    query_text = st.session_state.current_input
    k = st.session_state.search_k if st.session_state.search_k else 3  
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": k})

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
            docs = retriever.get_relevant_documents(resp_string)
            print("\n\n\n************ Docs in the context :", docs)
        #get answer using relevant documents and question
        result = st.session_state.document_chain.invoke({'input':resp_string, 
                                                'context':docs})
        

        #annotate the response with hallucination information
        # resp_sent, scores = check_sentence_hallucination(resp_string, docs, result, sample_size=3)
        resp_sent, scores = check_sentence_hallucination_cosine_similarity(docs, result)

        anno_result = ""
        for sent, score in zip(resp_sent, scores):
            if score < 0.5: #0 is no hallu, 1 is hallu / for cosine sim: 0 is hallu, 1 is
                sent = f":red-background[{sent}]"
            anno_result += sent 
        #result = anno_result
        print("Scores ***************", scores)
        print("RESULT ***************", result)
        print("RESULT ***************", anno_result)
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
def input_fields():
    
    st.session_state.llm = 'llama3.1'
    with st.sidebar:
        st.session_state.use_kb = st.toggle("Use Knowledge base.")
        k_list = [3,4,5,6,7]
        st.session_state.search_k = st.selectbox('No. of documents in context:', k_list)

        # st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
        # st.button("Submit documents", on_click=process_documents)

def get_session_gen_assist_chat_history():
    chat_list = st.session_state.messages_gen_assist 
    chat_history = []
    for conv in chat_list:
        if conv['speaker']=="user":
            chat_history.append(HumanMessage(content=conv['content']))
        if conv['speaker']=='AI':
            chat_history.append(AIMessage(content=conv['content']))
    return chat_history


# @st.cache_data(show_spinner=False)
# def process_documents():
#     if not st.session_state.source_docs:
#         st.warning(f"Please upload the documents first.")
#     else:

#         for source_doc in st.session_state.source_docs:
#             with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), 
#                                              prefix=source_doc.name.split('.')[0],
#                                              suffix='.pdf') as tmp_file:
#                 tmp_file.write(source_doc.read())
        
#         with st.spinner("Loading documents . . ."):
#             documents = load_documents()
#         #
#         for _file in TMP_DIR.iterdir():
#             temp_file = TMP_DIR.joinpath(_file)
#             temp_file.unlink()
#         #
#         with st.spinner("Parsing text . . ."):
#             texts = split_documents(documents)
#             #
#         k = st.session_state.search_k if st.session_state.search_k else 7
#         with st.spinner("Building database . . "):
#             st.session_state.vector_db =  create_vector_db(texts)   

def main():
    # page title
    st.set_page_config(page_title='Helper bot')
    st.title(
        'Helper bot'
    )
    input_fields()
    setup_llms_assistant()
    setup_llm_chains_assistant()
    st.session_state.vector_db = combine_vector_dbs(VECTOR_DB_PATHS['Public'], VECTOR_DB_PATHS['Private'])
    # App logic
    #uploaded_file = st.session_state.source_docs


    if "messages_gen_assist" not in st.session_state:
        st.session_state.messages_gen_assist = []

    st.chat_input(placeholder = 'Ask me anything: From writing emails to finding answers from documents. ', 
                    on_submit=query_chain,
                    key='current_input')

    with st.container(height=500):
        #display the chat history so far
        for msg in st.session_state.messages_gen_assist:
            st.chat_message(msg['speaker']).markdown(msg['content'])

    #display the documents in the context used to come up with the answer
    with st.container(height=500):
        if 'response_context' in st.session_state.keys():
            for doc in st.session_state.response_context:
                st.write(doc)

if __name__=='__main__':
  main()