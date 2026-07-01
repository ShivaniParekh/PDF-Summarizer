"""
Main Streamlit application file for the PDF Summarizer App.
This script sets up the user interface and orchestrates the RAG pipeline.
"""
import streamlit as st
from utils import summarizer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main execution function for the Streamlit web application.
    Constructs the UI, handles user PDF uploads, and displays the 
    generated summary using the underlying RAG architecture.
    """
    st.set_page_config(page_title="PDF Summarizer")
    st.title("PDF Summarizing App")
    st.write("Summarize your pdf files in just a few seconds.")
    st.divider()

    uploaded_pdf = st.file_uploader("Upload your PDF File",type="pdf")

    if uploaded_pdf is not None:
        # Check if a new file was uploaded by comparing file_id
        if "uploaded_pdf_id" not in st.session_state or st.session_state.uploaded_pdf_id != uploaded_pdf.file_id:
            st.session_state.pdf_bytes = uploaded_pdf.read()
            st.session_state.uploaded_pdf_id = uploaded_pdf.file_id
            st.session_state.pdf_name = uploaded_pdf.name # Store name for display if needed
            # No need to seek(0) here as we've read it once and stored the bytes.
    else:
        # Clear session state if no PDF is uploaded
        if "pdf_bytes" in st.session_state:
            del st.session_state.pdf_bytes
        if "uploaded_pdf_id" in st.session_state:
            del st.session_state.uploaded_pdf_id
        if "pdf_name" in st.session_state:
            del st.session_state.pdf_name
    
    query_input = st.text_input("Ask a question about your PDF:")
    
    col1, col2 = st.columns(2)
    with col1:
        submit_question = st.button("Submit Question")
    with col2:
        summarize_only = st.button("Summarize PDF")

    # Get credentials from environment variables loaded by dotenv.
    # Ensure the default model name is correctly prefixed.
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")
    # st.sidebar.write(f"DEBUG: Using Gemini Model: {gemini_model_name}") # For debugging .env loading and model selection
    
    # Use st.session_state.pdf_bytes instead of the uploaded_pdf object
    if "pdf_bytes" in st.session_state and st.session_state.pdf_bytes:
        pdf_bytes_to_process = st.session_state.pdf_bytes
        query_to_pass = ""
        if submit_question:
            if not query_input.strip():
                st.warning("Please enter a question or click 'Summarize PDF' for a summary.")
                return # Stop execution if no question is entered for a question submission
            query_to_pass = query_input.strip()
        elif summarize_only:
            query_to_pass = "Summarize the content of the uploaded PDF file in approximately 3-5 sentences."
        else: # No button clicked yet, or PDF uploaded but no action
            return # Do nothing until a button is clicked

        with st.spinner("Analyzing document and generating answer..."):
            # TRIGGER THE RAG PIPELINE
            response = summarizer(pdf_bytes_to_process, gemini_model_name, query_to_pass)
        st.subheader("Answer:")
        st.write(response)
    elif submit_question or summarize_only: # Buttons clicked but no PDF
        st.error("Please upload a PDF file first.")

if __name__ == '__main__':
    main()