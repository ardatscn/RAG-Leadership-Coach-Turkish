# RAG-Leadership-Coach-Turkish
In this project a Retrieval-Augmented Generation (RAG) model is created to generate a Leadership Coach in Turkish language.
The model was fed with the content gathered from the "Tecrübe Konuşuyor" playlist which is accessible from the link: https://www.youtube.com/playlist?list=PLCi3Q_-uGtdlCsFXHLDDHBSLyq4BkQ6gZ

To briefly explain the key points and steps in this repo:

  1- The data was prepared from the given YouTube playlist by the Whisper library from OpenAI. The transcripts were generated from each video
with high accuracy, enabling further processing such as text analysis and summarization.

  2- Recursive Text Splitting was used to chunk the data. Although there are more complex chunking methods based on semantic analysis, etc., 
this splitting method led to satisfactory results.

  3- For generating text embeddings, Google's text-embedding-004 model was employed. While numerous models are available for this task, the 
key considerations are selecting a bilingual model and ensuring reasonable API call limits for efficient processing.

  4- The FAISS vector database was utilized to store the generated text embeddings, offering a seamless integration with the existing LangChain
framework. Its ease of implementation and efficiency made it a compelling choice.

  5- The gemini-1.5-flash model was used as the LLM in the RAG implementation, with the temperature parameter set to 0 to minimize hallucinations.

  6- Custom prompts were provided alongside contextual information to guide the model's behavior effectively.

  7- Google Search (SerpApi) was used to trigger web searches when the required information was unavailable in the knowledge base.

  8- The Google Text-to-Speech (gTTS) library was used to convert the RAG model's outputs into speech.

  9- The code was deployed on Streamlit and is accessible at: https://rag-leadership-coach-turkish1.streamlit.app/

[Demo Videosunu İzle](https://github.com/ardatscn/RAG-Leadership-Coach-Turkish/blob/main/demo.mkv)

