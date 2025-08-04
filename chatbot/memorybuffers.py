import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("PERPLEXITY_API_KEY")
if not api_key:
    raise ValueError("PERPLEXITY_API_KEY not found. Make sure it's in your .env file.")

# Initialize model
llm = ChatPerplexity(
    api_key=api_key,
    model="sonar-pro"
)

# Create memory
memory = ConversationBufferMemory(return_messages=True)

# Create conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Streamlit UI
st.set_page_config(page_title="Perplexity Chatbot with Memory")
st.title("Chatbot with Memory")

# Initialize chat history in Streamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

input_text = st.text_input("Ask a question:")

if input_text:
    try:
        # Run the conversation chain
        response = conversation.predict(input=input_text)
        
        # Save to session chat history
        st.session_state.chat_history.append(("You", input_text))
        st.session_state.chat_history.append(("Bot", response))
        
        # Display conversation
        for sender, msg in st.session_state.chat_history:
            st.write(f"**{sender}:** {msg}")
    except Exception as e:
        st.error(f"Error: {e}")
