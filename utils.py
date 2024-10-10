
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pypdf import PdfReader

def process_text(text):
    # This function takes the text and splits it into chunks and
    #converts this chunks into embeddings to form a knowledge base.

    text_splitter= CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=20,length_function=len)
    chunks=text_splitter.split_text(text) #split text to chunks

    #loading the model to generate embeddings from chunks
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-miniLM-L6-v2')

    #Create a FAISS index from text chunks using the embeddings.
    knowledgeBase= FAISS.from_texts(chunks,embeddings)
    return knowledgeBase

def summarizer(pdf):
    #summarize the content
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=''

        for page in pdf_reader.pages:
            text+=page.extract_text() or ""

        knowlegdeBase= process_text(text)

        query="Summarize the content of the uploaded PDF file in approximately 3-5 sentences."

        if query:
            docs=knowlegdeBase.similarity_search(query)

            openAImodel= "gpt-3.5-turbo-16k"
            llm=ChatOpenAI(model=openAImodel,temperature=0.8)

            chain= load_qa_chain(llm,chain_type='stuff')
            response=chain.run(input_documents=docs,question=query)

            return response