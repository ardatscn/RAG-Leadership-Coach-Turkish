import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from serpapi import GoogleSearch
import requests
import time
import base64
import io
from gtts import gTTS

st.set_page_config(page_title="Leadership Coach Chatbot", layout="centered")

# Initialize Required API Keys (They are as Streamlib constants)
google_api_key = st.secrets.get("GOOGLE_API_KEY")
serp_api_key = st.secrets.get("SERPAPI_KEY")
os.environ["GOOGLE_API_KEY"] = google_api_key

## Load the Vector Embedding Model and Define the Chunking Method
@st.cache_resource     # Whole point of it to hinder streamlit from sending too many requests.
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

embeddings = load_embeddings()     # Get the vector embeddings from a model specified by model= parameter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)     # Chunk the data with specified chunk_size and chunk_overlap. They are hyperparameters that needs to be tuned.


## Uplaod The Data
scripts_dir = "https://api.github.com/repos/ardatscn/RAG-Leadership-Coach-Turkish/contents/video_scripts"     # Directory of the YouTube video scripts.
response = requests.get(scripts_dir, auth=("ardatscn", "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"))

all_texts = []
files = response.json() 
for file in files:     # Iterate through all scripts
  fname = file['name']
  raw_url = file["download_url"]     # Get the URL's
  
  file_response = requests.get(raw_url)
  file_content = file_response.text
  splitted_text = text_splitter.split_text(file_content)     # Chunk the data using Recursive Splitter
  all_texts.extend([(text, fname) for text in splitted_text])
    
txts, sources = zip(*all_texts)     # Combiene all of them

# Create and Store the Vector Embeddings
@st.cache_resource # Whole point of it to hinder streamlit from sending too many requests.
def load_vectors():
    vector_store = FAISS.from_texts(txts, embeddings, metadatas=[{"source": src} for src in sources])     # Embedding vectors from FAISS vector base
    return vector_store

vector_store = load_vectors()

## Use the desired LLM model for RAG implementation
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",  temperature=0)     # Temperature=0 hinders the model from making 'hallucinations'
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})     # Use similarity to select the vectors to our query

## Create a prompt template
prompt_template = (
    "Soruyu yanıtlamak için aşağıdaki talimatları kullanın. "
    "Yanıt Türkçe olmalıdır."
    "Cevap sağlanan bağlamda yoksa, Üzgünüm, cevap bulunamadı.. yanıtını verin. "
    "Referans aldığın kaynağı cevabında belirtmelisin."
    "Bir koçmuş ve tavsiye veriyormuş gibi konuş."
    "Context: {context}" # This place will retrieved from the user query.
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{input}"),
    ]
)


## Create the pipeline
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

def search_online(query):
    """
    Searches the web for additional information if not covered by knowledge-base
    """
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
    
@st.cache_data # Whole point of it to hinder streamlit from sending too many requests.
def search_online_c(query):
    return search_online(query)

def create_sound(text, lang="tr"):    # Generates the sound data
    tts = gTTS(text=text, lang=lang, slow=False)
    filename = "output.mp3"  # 🔹 Save file locally
    tts.save(filename)
    with open(filename, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes  

def sound_on(audio_data):    # Plays the generated sound
    b64 = base64.b64encode(audio_data).decode()    # Needed for streanlit implementation 
    autoplay_attr = "autoplay"
    md = f"""
    <audio style="display:none;" {autoplay_attr}>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

def RAG(query):
    response = chain.invoke({"input": query})
    answer = response["answer"]
    references = {doc.metadata["source"].replace(".txt", "") for doc in response["context"]}     # Remove the .txt extensions for cleaner references

    if "Üzgünüm, cevap bulunamadı" in answer:    # Search the web
        st.subheader("Eksik Veri! İşte İnternette Bulunan Sonuçlar:")
        result = search_online_c(query)
        all_snippets = ""
        for title, link, snippet in result:
            st.markdown(f"**[{title}]({link})**")
            st.write(f"{snippet}")
            all_snippets += snippet
        if sound:    # Sound on if checkbox
            audio_data = create_sound(all_snippets)
            sound_on(audio_data)
    else:
        st.success(answer)
        st.success(references)
        return answer
        
st.title("Leadership Coach")
st.write("Sorularınız 'Tecrübe Konuşuyor' YouTube Oynatım Listesinden Yanıtlanır.")
query = st.text_input("Sorunuzu Sorun:", placeholder="Örnek: Liderlerin ortak özellikleri nelerdir?")

sound = st.checkbox("Sound", value=True)
if st.button("Cevap Al"):
    if query:
        with st.spinner("Cevap Bekleniyor.."):
            answer = RAG(query)
            if answer and sound:
                with st.spinner("🔊 Generating speech..."):
                    audio_data = create_sound(answer)
                    sound_on(audio_data)     # Sound on if checkbox

            

