# PDF-Summarizer

PDF-Summarizer is a simple, lightweight web application built in Python that allows users to upload a PDF document and instantly get a concise, 3-5 sentence summary of its contents.

## How It Works
The application extracts text from uploaded PDF files, chunks the text, and processes it into dense vector embeddings using a local HuggingFace model. These embeddings are stored in a FAISS vector database. When a summary is requested, it performs a similarity search against the document's contents and passes the most relevant sections to an OpenAI Large Language Model (LLM) to generate a concise summary.
The application extracts text from uploaded PDF files, chunks the text, and processes it into dense vector embeddings using a local HuggingFace model. These embeddings are stored in a FAISS vector database.
When a user asks a question, the application performs a similarity search against the document's contents using the question, retrieves the most relevant sections, and passes them to a Google Gemini Large Language Model (LLM) to generate an answer.
When a summary is requested, a predefined summary prompt is used to perform a similarity search, retrieve relevant sections, and then passed to the Google Gemini LLM to generate a concise summary.

## Technology Stack
- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **Core AI Logic Pipeline**: [LangChain](https://python.langchain.com/)
- **Language Model (LLM)**: Google Gemini (`gemini-2.5-flash` by default) / [OpenAI](https://openai.com/) (`gpt-3.5-turbo-16k`) (OpenAI used in Previous Version)
- **Embeddings**: HuggingFace (`sentence-transformers/all-miniLM-L6-v2`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF (`pypdf` for text extraction)
