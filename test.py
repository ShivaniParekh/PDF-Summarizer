import streamlit as st
from utils import *
import os
import json

def main():
    st.set_page_config(page_title="PDF Summarizer")

    st.title("PDF Summarizing App")
    st.write("Summarize your pdf files in just a few seconds.")
    st.divider()

    pdf=st.file_uploader("Upload your PDF File",type="pdf")
    submit=st.button("Generate Summary")

    with open('config.json') as config_file:
        config = json.load(config_file)

    os.environ["OPENAI_API_KEY"]=config["API_KEY"]
    if submit:
        response=summarizer(pdf)
        st.subheader("Summary of file:")
        st.write(response)

if __name__ == '__main__':
    main()