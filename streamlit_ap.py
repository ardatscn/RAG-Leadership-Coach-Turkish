import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from elevenlabs.client import ElevenLabs
import requests


google_api_key = st.secrets.get("GOOGLE_API_KEY")
elevenlabs_api_key = st.secrets.get("ELEVENLABS_API_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key
embeddings =  GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
print('za')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Directory where transcripts of the YouTube contents are held
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

vector_store = FAISS.from_texts(txts, embeddings, metadatas=[{"source": src} for src in sources])
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",  temperature=0)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

system_prompt = (
    "Aşağıdaki bağlamları kullanarak soruyu yanıtlayın. "
    "Sağlanan bağlamdan en ayrıntılı şekilde soruyu yanıtlayın. "
    "Cevap sağlanan bağlamda yoksa, Üzgünüm cevabı bulamadım... demelisiniz. "
    "Yanıt Türkçe olmalıdır."
    "Referans aldığın kaynağı cevabında belirtmelisin."
    "Bir koçmuş gibi konuş."
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
    search = DuckDuckGoSearchResults(output_format="list")
    search_results = search.invoke(query)
    print(search_results)
    return search_results[2], search_results[2]

def query_rag(query):
    response = chain.invoke({"input": query})

    if isinstance(response, dict) and "answer" in response:
        answer = response["answer"]
        references = {doc.metadata["source"].replace(".txt", "") for doc in response["context"]}

        # Check if the answer contains "Üzgünüm, cevabı bulamadım..."
        if "Üzgünüm, cevabı bulamadım" in answer:
            print("\n📡 Bilgi eksik! Web'den ek kaynaklar aranıyor...\n")
            web_results, references = search_online(query)
            print(web_results)
            print(references)
            return web_results, references
        else:
          print("\n📜 Nihai Yanıt:\n", answer)
          print("\n References:")
          for ref in sorted(references):  # Convert set to sorted list for readability
            print(f"- {ref}")
          return answer, references
    else:
        return response, []

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.title("💬 RAG Chatbot with ElevenLabs TTS 🎙️")
st.write("**Sorularınızı sorun, yanıtlar hem metin hem de sesli olarak sağlansın!**")
query = st.text_input("📝 Sorunuzu yazın:", placeholder="Örnek: Arda Nehri nerededir?")

if st.button("🚀 Yanıt Al"):
    if query:
        with st.spinner("Yanıt oluşturuluyor..."):
            answer, references = query_rag(query)
            st.success(answer)
            st.success(references)

