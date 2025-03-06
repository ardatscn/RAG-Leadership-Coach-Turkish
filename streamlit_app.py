import streamlit as st
import os

st.title("Hello World")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
elevenlabs_api_key = st.secrets.get("ELEVENLABS_API_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key
embeddings =  GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
