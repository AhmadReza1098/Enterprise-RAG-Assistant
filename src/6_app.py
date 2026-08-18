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

# 2. Build the Webpage Interface
st.title("🏦 RBI Assistant AI")
st.write("Ask me anything about the RBI MSME guidelines!")

# 3. Load the AI Engine 
# (We use @st.cache_resource so it only loads the database once, making the chat super fast!)
@st.cache_resource
def load_ai():
    llm = ChatGroq(model_name="llama-3.1-8b-instant")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful, professional assistant answering questions about government documents.\n"
        "Answer the question based ONLY on the provided context. If the answer is not in the context, say you don't know.\n\n"
        "Context: {context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# Start the AI
chain = load_ai()

# 4. Create the Search Bar
user_question = st.text_input("Type your question here:")

# 5. Create the Submit Button
if st.button("Ask AI"):
    if user_question:
        # Show a loading spinner while the AI thinks
        with st.spinner("Searching the RBI documents..."):
            response = chain.invoke({"input": user_question})
            
            # Show the final answer on the screen!
            st.success("Found an answer!")
            st.write(response["answer"])
    else:
        st.warning("Please type a question first!")
