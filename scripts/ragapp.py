
import os, tempfile
from pathlib import Path
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

LOCAL_VECTOR_STORE_DIR = Path('../vectorstore')
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')

os.makedirs(TMP_DIR, exist_ok=True)
##############################  backend functions  ###############################

#for the vector store
def build_vector_db(file_list: list):
    #given a list of files builds a vector database from it
    return 0

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def create_vector_db(texts):
    vectordb = Chroma.from_documents(texts, embedding=HuggingFaceEmbeddings(),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    return vectordb

def build_retriever(dirpath):
    docs = load_documents(dirpath)
    texts = split_documents(docs)
    vector_db = create_vector_db(texts)
    retriever = vector_db.as_retriever(search_kwargs={'k': 7})
    return retriever


def setup_llm(llm_name:str):
    if llm_name=='llama3':
        return Ollama(model='llama3', system='You are a helpful question answering bot.')
    if llm_name=='claude':
        return AnthropicLLM(model='claude-2.1')



def setup_retriever_chain(retriever, llm, conversational=False):

    if conversational:
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
    else:
        prompt_get_answer_rag = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\\n\\n{context}. If you don't know the answer just say you dont know. Do not try to come up with something. Keep your answer brief and to the point."),
        ("user","{input}"),
        ])
        document_chain_rag =create_stuff_documents_chain(llm,prompt_get_answer_rag)
        rag_chain = create_retrieval_chain(retriever, document_chain_rag)

    return rag_chain

def query_chain():
    query_text = st.session_state.current_input
    k = st.session_state.search_k if st.session_state.search_k else 7
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={'k': k})
    st.session_state.rag_chain = setup_retriever_chain(retriever, 
                                                st.session_state.llm_model, 
                                                st.session_state.conversation_response)
      
    #save the query in the chat history
    st.session_state.messages.append({"speaker" : "user", "content": query_text})

    result = st.session_state.rag_chain.invoke({'input': query_text})

    st.session_state.response = result['answer']
    st.session_state.response_context = result['context']
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
            #
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
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
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={'k': k})
        st.session_state.rag_chain = setup_retriever_chain(retriever, 
                                                    st.session_state.llm_model, 
                                                    st.session_state.conversation_response)
      

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

    query_text = st.chat_input(placeholder = 'Enter query here ...', 
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