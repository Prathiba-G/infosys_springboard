import os
import time
from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
#from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import requests
from langchain groq import ChatGroq
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv
from langchain_chroma import Chroma
from src.function import download_huggingface_embedding, load_data_from_uploaded_pdf, load_data_from_url, load_data, text_split

def typing_animation(text, speed=0.1):
    placeholder = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.subheader(typed_text)
        time.sleep(speed)
    return placeholder

def main():
    load_dotenv()
    
    PINECONE_INDEX_NAME = "medical-chatbot"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    embeddings = download_huggingface_embedding()

    # Configure Streamlit page settings
    st.set_page_config(page_title="Medical-bot", layout="centered")
    
    if "animation_shown" not in st.session_state:
        st.session_state.animation_shown = False

    if not st.session_state.animation_shown:
        typing_animation("Hey, how can I help you today?")
        st.session_state.animation_shown = True

    st.title("Healthcare-bot")

    # Sidebar for data input options
    st.sidebar.header("Data Input Options")
    st.sidebar.write("Choose how you want to provide data for the chatbot:")
    st.sidebar.write("Upload Any Type of PDF and get your answer analysed by our Bot")

    if "selected_source" not in st.session_state:
        st.session_state.selected_source = None

    use_default_pdf = st.sidebar.checkbox(
        "Use Default PDF",
        key="default_pdf",
        on_change=lambda: st.session_state.update(selected_source="default_pdf" if st.session_state.default_pdf else None),
        disabled=st.session_state.selected_source is not None and st.session_state.selected_source != "default_pdf",
    )

    uploaded_pdf = st.sidebar.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="uploaded_pdf",
        disabled=st.session_state.selected_source is not None and st.session_state.selected_source != "uploaded_pdf",
    )

    uploaded_url = st.sidebar.text_input(
        "Enter a URL",
        key="uploaded_url",
        disabled=st.session_state.selected_source is not None and st.session_state.selected_source != "uploaded_url",
    )

    if use_default_pdf:
        st.session_state.selected_source = "default_pdf"
    elif uploaded_pdf is not None:
        st.session_state.selected_source = "uploaded_pdf"
    elif uploaded_url.strip():
        st.session_state.selected_source = "uploaded_url"
    else:
        st.session_state.selected_source = None

    if not st.session_state.selected_source:
        st.sidebar.warning("Please select a data source!")

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if "response" not in st.session_state:
        st.session_state.response = ""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Placeholder to display user choice or data
    if st.session_state.selected_source == "uploaded_pdf":
        st.success(f"PDF uploaded successfully! Processing {uploaded_pdf.name} file...")

        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        docs = load_data_from_uploaded_pdf("uploaded_file.pdf")
        doc_chunks = text_split(docs)
        docsearch = Chroma.from_documents(documents=doc_chunks, 
                                          embedding=embeddings,
                                          collection_name="PDF_database",
                                          persist_directory="./chroma_db_pdf")
    elif st.session_state.selected_source == "uploaded_url":
        st.success("URL provided: {}".format(uploaded_url))

        docs = load_data_from_url(uploaded_url)
        doc_chunks = text_split(docs)

        docsearch = Chroma.from_documents(
            documents=doc_chunks,
            embedding=embeddings,
            collection_name="URL_database",
            persist_directory="./chroma_db_url"
        )

    elif st.session_state.selected_source == "default_pdf":
        st.success("Using GALE ENCYCLOPEDIA OF MEDICINE data!")

        try:
            docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
            st.success("Index loaded successfully!")
        except Exception as e:
            st.error("Error loading index: {}".format(e))

    else:
        st.info("Please upload a file, enter a URL, or select default data to proceed.")
        st.stop()

    prompt_template = """
    Use the given information context to give appropriate answer for the user's question.
    If you don't know the answer, just say that you know the answer, but don't make up an answer.
    Context: {context}
    Question: {question}
    Only return the appropriate answer and nothing else.
    Helpful answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    llm =Ollama(model="llama2-7b")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=docsearch.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    ) 

    st.divider()
    user_input = st.text_input(
        "Ask something to the chatbot:",
        value=st.session_state.user_input,
        placeholder="Type your question here..."
    )

    if not user_input.strip():
        st.info("Please enter a question to get a response from the chatbot.")

    submit_button_disabled = not user_input.strip()
    submit_button = st.button(
        "Submit", 
        disabled=submit_button_disabled, 
        key="submit_query",
        help="Click to submit your query"
    )

    if submit_button:
        if user_input.strip():
            with st.spinner("Processing your query... Please wait."):
                try:
                    result = qa.invoke(user_input)
                    response = result['result']

                    st.session_state.chat_history.append({"question": user_input, "answer": response})
                    st.session_state.response = response

                    st.session_state.user_input = ""
                except Exception as e:
                    st.session_state.response = f"Error generating response: {e}"
        else:
            st.warning("Please enter a query to get a response!")

    if st.session_state.response:
        st.subheader(f"**Response:** {st.session_state.response}")

    if len(st.session_state.chat_history) > 0:
        with st.expander("Chat History"):
            for chat in st.session_state.chat_history:
                st.markdown(
                    f"""
                    <div style="background-color: #f0f0f0; color: #000; padding: 10px 15px; border-radius: 10px; 
                    margin-bottom: 20px; margin-left: auto; max-width: 70%; word-wrap: break-word; text-align: left;">
                        {chat['question']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div style="background-color: #4CAF50; color: #fff; padding: 10px 15px; border-radius: 10px; 
                    margin-bottom: 10px; max-width: 70%; word-wrap: break-word; text-align: left;">
                        {chat['answer']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

if __name__ == "__main__":
    main()
