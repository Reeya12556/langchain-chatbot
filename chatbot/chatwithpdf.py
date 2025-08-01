from langchain_community.chat_models import ChatPerplexity
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("PERPLEXITY_API_KEY")

if not api_key:
    raise ValueError("PERPLEXITY_API_KEY not found in .env file")

# Streamlit 
st.set_page_config(page_title="Chat withPDF")
st.title(" PDF Chatbot ")

uploaded_file = st.file_uploader("C:\Users\Laptop\Desktop\Langchain\pdffolder\Introduction to AI.pdf", type="pdf")
query = st.text_input("Ask a question about the PDF:")

if uploaded_file:
    # Load and split PDF
    loader = PyPDFLoader(uploaded_file.name)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)

    # Convert chunks to embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Set up Perplexity LLM
    llm = ChatPerplexity(api_key=api_key, model="sonar-pro")

    # Retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True  
    )

    if query:
        with st.spinner("Thinking.."):
            result = qa_chain.invoke({"query": query})
            st.subheader(" Answer")
            st.write(result["result"])

            # Optional: Show source documents
            with st.expander("Sources"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)
