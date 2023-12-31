import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from llama_index import LLMPredictor
from langchain import HuggingFaceHub
from streamlit.components.v1 import html
from pathlib import Path
from time import sleep
import random
import string
import sys
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Cheers! Open AI Doc-Chat Assistant", layout="wide")
st.subheader("Open AI Doc-Chat Assistant: Life Enhancing with AI!")

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
repo_id=os.getenv("repo_id")
model_name=os.getenv("model_name")

documents=[]
wechat_image= "WeChatCode.jpg"

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))  
random_string = generate_random_string(20)
directory_path=random_string

st.sidebar.markdown(
    """
    <style>
    .blue-underline {
        text-decoration: bold;
        color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
        }
    </style>
    """, unsafe_allow_html=True
)

question = st.text_input("Enter your query here:")
display_output_text = st.checkbox("Check AI Repsonse", key="key_checkbox", help="Check me to get AI Response.")

with st.sidebar:    
    pdf_files = st.file_uploader("Upload file and start AI Doc-Chat.", type=['pdf'], accept_multiple_files=True)
    st.write("Disclaimer: This app is for information purpose only. NO liability could be claimed against whoever associated with this app in any manner. User should consult a qualified legal professional for legal advice.")
    st.sidebar.markdown("Contact: [aichat101@foxmail.com](mailto:aichat101@foxmail.com)")
    st.sidebar.markdown('WeChat: <span class="blue-underline">pat2win</span>, or scan the code below.', unsafe_allow_html=True)
    st.image(wechat_image)
    st.sidebar.markdown('<span class="blue-underline">Life Enhancing with AI.</span>', unsafe_allow_html=True)      
    st.subheader("Enjoy chatting!")
    if pdf_files:
        os.makedirs(directory_path)
        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file.name)
            with open(file_path, 'wb') as f:
                f.write(pdf_file.read())
            st.success(f"File '{pdf_file.name}' saved successfully.")
        documents = SimpleDirectoryReader(directory_path).load_data()    
    else:
        print("waiting for path creation.")
        sys.exit()

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":1024,
                                   "max_new_tokens":5632, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155}) 

llm_predictor = LLMPredictor(llm)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

new_index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
)

if question !="" and not question.strip().isspace() and not question == "" and not question.strip() == "" and not question.isspace():
    if display_output_text==True:
      with st.spinner("AI Thinking...Please wait a while to Cheers!"):
        new_index.storage_context.persist("directory_path")
        storage_context = StorageContext.from_defaults(persist_dir="directory_path")
        loadedindex = load_index_from_storage(storage_context=storage_context, service_context=service_context)
        query_engine = loadedindex.as_query_engine() 
        initial_response = query_engine.query(question)
        #temp_ai_response=str(initial_response)
        #final_ai_response=temp_ai_response.partition('<|end|>')[0]
        st.write("AI Response:\n\n"+str(initial_response))
    else:
        print("Check the Checkbox to get AI Response.")
        sys.exit()          
else:
    print("Please enter your question first.")
    st.stop()
