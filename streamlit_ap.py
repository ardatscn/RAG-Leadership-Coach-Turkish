import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from elevenlabs.client import ElevenLabs
from serpapi import GoogleSearch
import requests
import time
import base64
import io

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
elevenlabs_api_key = st.secrets.get("ELEVENLABS_API_KEY")
serp_api_key = st.secrets.get("SERPAPI_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key

@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

embeddings = load_embeddings()  # Cached and will not reload on button click

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#Directory
scripts_dir = "https://api.github.com/repos/ardatscn/RAG-Leadership-Coach-Turkish/contents/video_scripts"
response = requests.get(scripts_dir, auth=("ardatscn", "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"))

all_texts = []
files = response.json() 
a = 0
for file in files:
  fname = file['name']
  raw_url = file["download_url"]  # Get the raw URL of the file
  
  # Read the content of the file
  file_response = requests.get(raw_url)
  file_content = file_response.text  # Convert response to text
  splitted_text = text_splitter.split_text(file_content)
  all_texts.extend([(text, fname) for text in splitted_text])
    
txts, sources = zip(*all_texts)

@st.cache_resource
def load_vectors():
    vector_store = FAISS.from_texts(txts, embeddings, metadatas=[{"source": src} for src in sources])
    return vector_store

vector_store = load_vectors()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",  temperature=0)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

system_prompt = (
    "Soruyu yanıtlamak için aşağıdaki talimatları kullanın. "
    "Yanıt Türkçe olmalıdır."
    "Cevap sağlanan bağlamda yoksa, Üzgünüm, cevap bulunamadı.. yanıtını verin. "
    "Referans aldığın kaynağı cevabında belirtmelisin."
    "Bir koçmuş ve tavsiye veriyormuş gibi konuş."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

def search_online(query):
    params = {
        "q": query,
        "hl": "tr",
        "gl": "tr",
        "api_key": serp_api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    search_results = results.get("organic_results", [])
    return [(res.get("title", "No Title"), res.get("link", "#"), res.get("snippet", "No Snippet")) for res in search_results[:5]]
    
@st.cache_data
def search_online_cached(query):
    return search_online(query)

client = ElevenLabs(api_key=elevenlabs_api_key)

def generate_voice(text):
    """Generates speech from text using ElevenLabs API and returns audio bytes."""
    audio_stream = client.text_to_speech.convert_as_stream(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2"
    )
    
    # ✅ Convert Generator to Bytes (Avoid Writing to Disk)
    audio_bytes = b"".join(audio_stream)

    return audio_bytes

# 🔹 Function to Play Audio in Streamlit
def play_audio(audio_data):
    """Embeds an audio player in Streamlit."""
    b64 = base64.b64encode(audio_data).decode()
    md = f"""
    <audio controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)


def query_rag(query):
    response = chain.invoke({"input": query})
    answer = response["answer"]
    references = {doc.metadata["source"].replace(".txt", "") for doc in response["context"]}

    if "Üzgünüm, cevap bulunamadı" in answer:
        st.subheader("Eksik Veri! İşte İnternette Bulunan Sonuçlar:")
        result = search_online_cached(query)
        for title, link, snippet in result:
            st.markdown(f"**[{title}]({link})**")
            st.write(f"{snippet}")

    else:
        st.success(answer)
        st.success(references)

st.title("Leadership Coach")
st.write("Sorularınız 'Tecrübe Konuşuyor' YouTube Oynatım Listesinden Yanıtlanır.")
query = st.text_input("Sorunuzu Sorun:", placeholder="Örnek: Liderlerin ortak özellikleri nelerdir?")


if st.button("Cevap Al"):
    if query:
        with st.spinner("Cevap Bekleniyor.."):
            query_rag(query)
            with st.spinner("🔊 Generating speech..."):
                audio_data = generate_voice(final_answer)
                play_audio(audio_data)
            

