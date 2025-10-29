import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    return chain

def user_input(user_question, k=4):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if not os.path.exists("./chroma_db"):
        return "Please upload and process PDF files first!"
    
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    docs = vector_store.similarity_search(user_question, k=k)
    
    chain = get_conversational_chain()
    
    response = chain.invoke({
        "context": docs,
        "question": user_question
    })
    
    return response

def main():
    st.set_page_config(
        page_title="PDF Chat Assistant",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main {
            background-color: #f7f7f8;
        }
        .stChatMessage[data-testid="user-message"] {
            background-color: white !important;
            color: black !important;
        }

        .stChatMessage[data-testid="assistant-message"] {
            background-color: #f7f7f8 !important;
            color: black !important;
        }
        .stChatMessage p, .stChatMessage div {
            color: black !important;
        }
        .stChatInput > div {
            background-color: white !important;
        }
        .stChatInput input {
            background-color: white !important;
            color: black !important;
        }
        .stChatInput textarea {
            background-color: white !important;
            color: black !important;
        }
        div[data-testid="stSidebarNav"] {
            background-color: #202123;
        }
        .sidebar .sidebar-content {
            background-color: #202123;
        }
        h1 {
            color: #2c3e50;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Sidebar
    with st.sidebar:      
        pdf_docs = st.file_uploader(
            "Upload PDF Files", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Process Documents", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.pdf_processed = True
                    st.success("Documents processed successfully!")
            else:
                st.error("Please upload PDF files first!")
        
        st.markdown("---")
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.title("PDF Chat Assistant")
    st.markdown("Ask questions about your PDF documents")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        if not st.session_state.pdf_processed and not os.path.exists("./chroma_db"):
            with st.chat_message("assistant"):
                st.warning("Please upload and process PDF files first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()