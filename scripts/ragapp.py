
import os, tempfile
from pathlib import Path
os.path.join('..')
os.environ['ANTHROPIC_API_KEY'] = "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"
import streamlit as st
import pytesseract

from langchain.vectorstores import Chroma

#import for llms
from langchain_community.llms import Ollama
from langchain_anthropic import AnthropicLLM

#import for embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma 
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, ConversationChain, LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from prompts.prompt_template import *
from utils.utils import *
import ipdb 

import warnings
warnings.filterwarnings("ignore")

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import torch
import spacy 
from annotated_text import annotated_text

nlp = spacy.load("en_core_web_sm")


LOCAL_VECTOR_STORE_DIR = Path('../vectorstore')

PERSIST_DIRECTORY = Path('../vectorstore_test')
TMP_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

##############################  backend functions  ###############################

#for the vector store
def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    return texts

def create_vector_db(texts):
    #vectordb = Chroma.from_documents(texts, embedding=HuggingFaceEmbeddings())
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY.as_posix(), embedding_function=HuggingFaceEmbeddings())
    #vectordb.persist()
    return vectordb


def setup_llms():

    if st.session_state.llm=='llama3':
        st.session_state.llm_model_chat = Ollama(model='llama3', system='You are a helpful question answering bot.')
        st.session_state.llm_model_instruct = Ollama(model='llama3', system="You are an LLM that is excellent at following instructions")
    
    if st.session_state.llm=='claude':
        return AnthropicLLM(model='claude-2.1')
    

def setup_llm_chains():

    #build the conversation chain
    conv_prompt = PromptTemplate(input_variables=['input', 'history'], template=CONV_PROMPT_TEMPLATE)
    st.session_state.conv_chain = LLMChain(llm=st.session_state.llm_model_chat, prompt=conv_prompt, output_key='answer')


    #build the retriever chain 

    # prompt_search_query = ChatPromptTemplate.from_messages([
    # MessagesPlaceholder(variable_name="chat_history"),
    # ("user","{input}"),
    # ("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    # ])
    # retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)
    
    #build the rephrase chain 
    st.session_state.rephrase_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=RETRIEVE_REPHRASE_PROMPT)


    #build the document chain
    st.session_state.document_chain=create_stuff_documents_chain(st.session_state.llm_model_chat, DOCUMENT_CHAIN_PROMPT)

    #build the router chain
    router_prompt = PromptTemplate(
        input_variables=["input"], template=ROUTER_PROMPT_TEMPLATE
    )
    st.session_state.router_chain = LLMChain(llm=st.session_state.llm_model_instruct, prompt=router_prompt, output_key='answer')


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


def query_chain():
    query_text = st.session_state.current_input
    sources = set([val['source'] for val in st.session_state.vector_db.get()['metadatas']])
    k = st.session_state.search_k if st.session_state.search_k else 3  
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": k})

    #use chains

    #check if retrieval is required
    resp = st.session_state.router_chain.invoke({'input': query_text})
    is_qa = get_key_val_from_llm_json_string(resp['answer'], 'response')
    
    input_dict = {'input': query_text, 'chat_history': get_session_chat_history()}
    
    if is_qa.strip().lower()=='yes':

        #rephrase question using history
        resp = st.session_state.rephrase_chain.invoke(input_dict)
        resp_string = get_key_val_from_llm_json_string(resp['text'], 'rephrased_input')
        print("**********Rephrased input :", resp_string)
        #use response to retrieve relevant documents 
        docs = retriever.get_relevant_documents(resp_string)

        #get answer using relevant documents and question
        result = st.session_state.document_chain.invoke({'input':resp_string, 
                                                'context':docs})

        resp_sent, scores = check_sentence_hallucination(resp_string, docs, result, sample_size=3)
        anno_result = ""
        for sent, score in zip(resp_sent, scores):
            if score < 0.2: #0 is no hallu, 1 is hallu
                sent = f":red-background[{sent}]"
            anno_result += sent 
        #result = anno_result
        print("Scores ***************", scores)
        print("RESULT ***************", result)
        print("RESULT ***************", anno_result)

        st.session_state.response = anno_result
        st.session_state.response_context = docs

    else:
        result = st.session_state.conv_chain.invoke(input_dict)
        anno_result = result['answer']
        st.session_state.response = result
        st.session_state.response_context = "NO CONTEXT FOR REGULAR CHAT"    
    
    #save the query in the chat history
    st.session_state.messages.append({"speaker" : "user", "content": query_text})
    
    #annotate the response with hallucination information
    st.session_state.messages.append({"speaker" : "AI",
                                       "content": anno_result})
  

################################  front end functions  ################################
def input_fields():
    
    st.session_state.llm = 'llama3'
    with st.sidebar:

        st.session_state.llm = st.selectbox('Select an LLM', ['llama3', 'claude'], index=0)
        st.session_state.conversation_response = st.toggle("Enable conversaion", value=False)
        k_list = [3,4,5,6,7]
        st.session_state.search_k = st.selectbox('No. of documents in context:', k_list)

    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)

def get_session_chat_history():
    chat_list = st.session_state.messages 
    chat_history = []
    for conv in chat_list:
        if conv['speaker']=="user":
            chat_history.append(HumanMessage(content=conv['content']))
        if conv['speaker']=='AI':
            chat_history.append(AIMessage(content=conv['content']))
    return chat_history

@st.cache_data
def process_documents():
    if not st.session_state.source_docs:
        st.warning(f"Please upload the documents first.")
    else:

        for source_doc in st.session_state.source_docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), 
                                             prefix=source_doc.name.split('.')[0],
                                             suffix='.pdf') as tmp_file:
                tmp_file.write(source_doc.read())
        
        documents = load_documents()
        #
        for _file in TMP_DIR.iterdir():
            temp_file = TMP_DIR.joinpath(_file)
            temp_file.unlink()
        #
        texts = split_documents(documents)
            #
        k = st.session_state.search_k if st.session_state.search_k else 7
        st.session_state.vector_db =  create_vector_db(texts)   

def main():
    # page title
    st.set_page_config(page_title='Helper bot')
    st.title(
        'Helper bot'
    )
    input_fields()
    setup_llms()
    setup_llm_chains()

    st.button("Submit documents", on_click=process_documents)


    # question_list = [
    #     'How many rows are there?',
    #     'What is the range of values for MolWt with logS greater than 0?',
    #     'How many rows have MolLogP value greater than 0.',
    #     'Other']
    # query_text = st.selectbox('Select an example query:', question_list)

    # App logic
    uploaded_file = st.session_state.source_docs

    st.chat_input(placeholder = 'Enter query here ...', 
                            disabled=not uploaded_file, 
                            on_submit=query_chain,
                            key='current_input')
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.container(height=500):
        #display the chat history so far
        for msg in st.session_state.messages:
            st.chat_message(msg['speaker']).markdown(msg['content'])

    #display the documents in the context used to come up with the answer
    with st.container(height=200):
        if 'response_context' in st.session_state.keys():
            for doc in st.session_state.response_context:
                st.write(doc)

if __name__=='__main__':
  main()