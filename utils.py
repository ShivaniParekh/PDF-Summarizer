"""
Utility module containing the core Retrieval-Augmented Generation (RAG) pipeline logic.
Provides functions for chunking PDFs, creating vector embeddings, and generating summaries via Google Gemini.
"""
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
            create_stuff_documents_chain,
        )
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader
import streamlit as st

@st.cache_resource(show_spinner="Building Vector Database...")
def process_text(text):
    """
    Ingests raw text and prepares it for the Vector Database.
    
    This function handles the first stage of the RAG pipeline:
    1. Chunking: Splits large text into smaller segments.
    2. Embedding: Converts text chunks into numerical vectors.
    3. Vector DB: Stores embeddings in a searchable FAISS index.

    Args:
        text (str): The raw text extracted from the PDF.

    Returns:
        FAISS: A populated FAISS vector store index ready for similarity search.
    """

    # --- RAG STAGE 1a: CHUNKING ---
    # We break the large PDF text into smaller, 1000-character chunks so the LLM can process them
    text_splitter= CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=20,length_function=len)
    chunks=text_splitter.split_text(text) #split text to chunks

    # --- RAG STAGE 1b: EMBEDDING ---
    # We convert those English chunks into numerical vectors using a local HuggingFace model
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-miniLM-L6-v2')

    # --- RAG STAGE 1c: VECTOR DATABASE ---
    # We store those vectors in a FAISS database so they can be quickly searched
    knowledgeBase= FAISS.from_texts(chunks,embeddings)
    return knowledgeBase

@st.cache_data(show_spinner="Extracting Text from PDF...")
def extract_text_from_pdf(pdf_bytes):
    # We pass the bytes to the reader, extract text, and return the string
    import io
    pdf_reader=PdfReader(io.BytesIO(pdf_bytes))
    text=''
    for page in pdf_reader.pages:
        text+=page.extract_text() or ""
    return text

def summarizer(pdf_bytes, gemini_model_name, query):
    """
    Orchestrates the retrieval and generation stages of the RAG pipeline.
    
    Given PDF bytes, this function extracts the text, builds the vector database,
    retrieves the most relevant chunks based on the user's query, and prompts the
    Google Gemini model to generate a final answer.

    Args:
        pdf_bytes (bytes): The content of the uploaded PDF file.
        gemini_model_name (str): The specific Google Gemini model to use.
        query (str): The user's question about the PDF.

    Returns:
        str: The generated markdown/text answer.
    """
    if pdf_bytes is not None: # Check if bytes are provided
        text = extract_text_from_pdf(pdf_bytes)
        
        # Edge case handling: Empty or Scanned PDFs
        if not text.strip():
            return "Error: No readable text found in this PDF. It might be a scanned image."

        # Caching optimization: Build vector DB only if text changes
        knowlegdeBase = process_text(text)

        llm=ChatGoogleGenerativeAI(model=gemini_model_name,temperature=0.8)

        if query:
            # --- RAG STAGE 2: RETRIEVAL ---
            docs=knowlegdeBase.similarity_search(query)
            
            # --- RAG STAGE 3: AUGMENTED GENERATION ---
            # Define a simple prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Answer:"""
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            
            # Use the modern LCEL chain creation instead of load_qa_chain
            chain = create_stuff_documents_chain(llm, PROMPT)
            
            # Invoke the chain
            response = chain.invoke({"context": docs, "question": query})
        return response