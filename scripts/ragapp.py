
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
import ipdb 

LOCAL_VECTOR_STORE_DIR = Path('../vectorstore')
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
    vectordb = Chroma.from_documents(texts, embedding=HuggingFaceEmbeddings())
                                    # persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    #vectordb.persist()
    return vectordb


def setup_llm(llm_name:str):
    if llm_name=='llama3':
        return Ollama(model='llama3', system='You are a helpful question answering bot.')
    if llm_name=='claude':
        return AnthropicLLM(model='claude-2.1')
    


def parse_resp(response_string):
    '''
    Parse the response to get the one word answer.
    '''
    full_response = response_string.strip().split('{')[1].split('}')[0]
    one_word_response = full_response.split(",")[0]
    word = one_word_response.split(':')[1].strip('" "')
    return word

def setup_llm_chains(retriever, llm):

    #build the conversation chain
    conv_prompt = PromptTemplate(input_variables=['input', 'history'], template=CONV_PROMPT_TEMPLATE)
    conv_chain = LLMChain(llm=llm, prompt=conv_prompt, output_key='answer')


    #build the retriever chain 
    prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    ("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)


    prompt_get_answer = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\\n\\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    ])
    document_chain=create_stuff_documents_chain(llm,prompt_get_answer)
    rag_chain = create_retrieval_chain(retriever_chain, document_chain)


    #build the router chain
    router_prompt = PromptTemplate(
        input_variables=["input"], template=ROUTER_PROMPT_TEMPLATE
    )
    router_chain = LLMChain(llm=llm, prompt=router_prompt, output_key='answer')

    #setup the session variables
    st.session_state.conv_chain = conv_chain
    st.session_state.rag_chain = rag_chain 
    st.session_state.router_chain = router_chain 



def query_chain():
    query_text = st.session_state.current_input
    k = st.session_state.search_k if st.session_state.search_k else 7
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={'k': k})

    sources = set([val['source'] for val in st.session_state.vector_db.get()['metadatas']])
    ipdb.set_trace()
    setup_llm_chains(retriever, st.session_state.llm_model)
      

    #use chains
    resp = st.session_state.router_chain.invoke({'input': query_text})
    is_qa = parse_resp(resp['answer'])
    
    input_dict = {'input': query_text, 'chat_history': get_session_chat_history()}
    if is_qa.strip().lower()=='yes':
        result = st.session_state.rag_chain.invoke(input_dict)
        st.session_state.response = result['answer']
        st.session_state.response_context = result['context']
    else:
        result = st.session_state.conv_chain.invoke(input_dict)
        st.session_state.response = result['answer']
        st.session_state.response_context = "NO CONTEXT FOR REGULAR CHAT"    
    
    #save the query in the chat history
    st.session_state.messages.append({"speaker" : "user", "content": query_text})
    st.session_state.messages.append({"speaker" : "AI",
                                       "content": result['answer']})
  

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
            ipdb.set_trace()
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), 
                                             prefix=source_doc.name.split('.')[0],
                                             suffix='.pdf') as tmp_file:
                tmp_file.write(source_doc.read())
        
        documents = load_documents()
        print("**********\n\n\n")
        print(len(documents))
        print("*****************")
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
    st.set_page_config(page_title='🦜🔗 Ask the Data App')
    st.title(
        '🦜🔗 Ask the Data App'
    )
    input_fields()
    st.button("Submit documents", on_click=process_documents)
    st.session_state.llm_model = setup_llm(st.session_state.llm)


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
            st.chat_message(msg['speaker']).write(msg['content'])

    #display the documents in the context used to come up with the answer
    with st.container(height=200):
        if 'response_context' in st.session_state.keys():
            for doc in st.session_state.response_context:
                st.write(doc)

if __name__=='__main__':
  main()