import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup API Key
os.environ["GROQ_API_KEY"] = "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"
DB_PATH = "data/vector_db"

# Page config to make the tab look nice
st.set_page_config(page_title="RBI Assistant", page_icon="🏦")
st.title("🏦 RBI Assistant AI")
st.write("Ask me anything about the RBI MSME guidelines!")

# 2. Load the AI Engine 
@st.cache_resource
def load_ai():
    llm = ChatGroq(model_name="llama-3.1-8b-instant")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert financial and technical assistant. \n"
        "Your goal is to answer the user's question based ONLY on the provided context.\n\n"
        "INSTRUCTIONS:\n"
        "1. First, think step-by-step about how the context relates to the question.\n"
        "2. Write out your thinking process clearly.\n"
        "3. Finally, provide your official answer.\n"
        "4. If the answer is not in the context, say 'I cannot find this in the documents.'\n\n"
        "Context: {context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

chain = load_ai()

# 3. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display all previous messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Create the Chat Input Bar at the bottom
if prompt := st.chat_input("Type your question here..."):
    
    # Show the user's message on the screen and save it to history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show the AI's response on the screen and save it to history
    with st.chat_message("assistant"):
        with st.spinner("Searching the RBI documents..."):
            response = chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
