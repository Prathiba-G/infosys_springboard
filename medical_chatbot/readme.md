This project is a Retrieval-Augmented Generation (RAG) Healthcare Chatbot that provides accurate and context-aware medical information using an LLM-based approach. It integrates Pinecone for vector indexing, Hugging Face embeddings for text representation, Groq API as the LLM model, and ChromaDB for additional vector storage.

<h2>Features</h2>
Medical Question Answering: Uses RAG to retrieve relevant medical documents before generating responses.
Pinecone Indexing: Efficient storage and retrieval of vectorized text.
Hugging Face Embeddings: Converts textual data into high-quality vector representations.
Groq API (LLM): Generates responses based on retrieved context.
ChromaDB: Alternative vector database for enhanced retrieval.
<h2>Tech Stack</h2>
Language Model: llama2-7b using Groq API
Vector Storage: Pinecone, ChromaDB
Embeddings Model: Hugging Face Sentence Transformers
Backend: Python 
Frontend : Streamlit
