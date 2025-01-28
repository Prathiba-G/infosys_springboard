import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "yolov9pdf"
index = pc.Index(index_name)

# Load Hugging Face model for embeddings
embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

# Function to get LLM inference
def get_llm_hf_inference(model_id="google/flan-t5-base", max_new_tokens=128, temperature=0.1):
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token=os.getenv("HF_TOKEN")
    )
    return llm

# Function to retrieve context from Pinecone
def retrieve_from_pinecone(user_query):
    pinecone_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    context = pinecone_store.similarity_search(user_query)[:5]
    return context

# Function to generate a response
def get_response(user_query):
    context = retrieve_from_pinecone(user_query)
    llm = get_llm_hf_inference()
    
    template = """Answer the question below according to your knowledge in a way that will be helpful to users.
    Context: {context}
    User question: {user_question}"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser(output_key='content')
    
    response = chain.invoke(input={"context": context, "user_question": user_query})
    return response

# Streamlit app configuration
st.set_page_config(page_title="YOLOv9 PDF Chatbot", page_icon="🤖")
st.title("YOLOv9 PDF Chatbot")

# Chat history initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! How can I assist you with the YOLOv9 PDF today?")]

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)
    
    with st.chat_message("AI"):
        response = get_response(user_query)
        st.write(response)
        st.session_state.chat_history.append(AIMessage(content=response))
