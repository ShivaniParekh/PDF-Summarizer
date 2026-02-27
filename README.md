# PDF-Summarizer

PDF-Summarizer is a simple, lightweight web application built in Python that allows users to upload a PDF document and instantly get a concise, 3-5 sentence summary of its contents.

## How It Works
The application extracts text from uploaded PDF files, chunks the text, and processes it into dense vector embeddings using a local HuggingFace model. These embeddings are stored in a FAISS vector database. When a summary is requested, it performs a similarity search against the document's contents and passes the most relevant sections to an OpenAI Large Language Model (LLM) to generate a concise summary.

## Technology Stack
- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **Core AI Logic Pipeline**: [LangChain](https://python.langchain.com/)
- **Language Model (LLM)**: [OpenAI](https://openai.com/) (`gpt-3.5-turbo-16k`)
- **Embeddings**: HuggingFace (`sentence-transformers/all-miniLM-L6-v2`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF (`pypdf` for text extraction)

## Setup
1. Clone the repository.
2. Install the necessary dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your OpenAI API key to the `config.json` file. Note that this file should not be committed to version control.
4. Run the application:
   ```bash
   streamlit run test.py
   ```