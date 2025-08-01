from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API key 
api_key = os.getenv("PERPLEXITY_API_KEY")
if not api_key:
    raise ValueError("PERPLEXITY_API_KEY not found. Make sure it's in your .env file.")

# Set up the model
llm = ChatPerplexity(
    api_key=api_key,
    model="sonar-pro"  
)

#prompt template
prompt = ChatPromptTemplate.from_template("Answer the question: {question}")

# Set up the output parser
output_parser = StrOutputParser()

# Create the chain
chain = prompt | llm | output_parser

# Streamlit app
st.set_page_config(page_title="Perplexity Chatbot")
st.title(" Ask Me Anything")

input_text = st.text_input("Enter your question here:")

if input_text:
    try:
        response = chain.invoke({"question": input_text})
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
