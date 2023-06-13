import os
import streamlit as st
import json
import time
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_lottie import st_lottie

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

st.set_page_config(page_title="Ask IDMPete", page_icon="man")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
lottie_hello = load_lottiefile("IDMPete.json")
    
col1, col2 = st.columns(2)

with col1:
   st.title("Ask IDMPete")
   intro_text_1 = '<p style="font-size: 22px;">Hi👋 I am IDMPete and I am here to help you. I have read chapter 2 of the ISO IDMP implementation guide and I will try to answer all your questions about it.</p>'
   st.markdown(intro_text_1, unsafe_allow_html=True)
   intro_text_2 = '<p style="font-size: 22px;">Give it a try and drop you question for me below!</p>'
   st.markdown(intro_text_2, unsafe_allow_html=True)

with col2:
    st_lottie(lottie_hello)

st.subheader("Enter your question")

def get_text():
    question = st.text_area(label="Enter your question", placeholder="Type your question here...", key="user_question", label_visibility="collapsed")
    return question

user_question = get_text()

st.subheader("IDMPete says...")

if user_question:
    with st.spinner('...great question!'):
        pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
        )

        docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")

        docs = docsearch.similarity_search(user_question)
        answer = chain.run(input_documents=docs, question=user_question)

    st.write(answer)

