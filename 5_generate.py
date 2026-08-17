import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ⚠️ PASTE YOUR GROQ API KEY HERE ⚠️
os.environ["GROQ_API_KEY"] = "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"

DB_PATH = "data/vector_db"

def ask_ai(question):
    print("Waking up the Groq AI...")
    
    # 1. Load the AI Model (Using the new Llama 3.1)
    llm = ChatGroq(model_name="llama-3.1-8b-instant") 
    
    # 2. Load Your Vector Database
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
    
    # Fetch top 2 most relevant chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    
    # 3. Give the AI its instructions
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful, professional assistant answering questions about government documents.\n"
        "Answer the question based ONLY on the provided context. If the answer is not in the context, say you don't know.\n\n"
        "Context: {context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    )
    
    # 4. Connect everything together
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print(f"\nThinking about: '{question}'...\n")
    
    # 5. Ask the question!
    response = retrieval_chain.invoke({"input": question})
    
    print("--- Final AI Answer ---")
    print(response["answer"])

if __name__ == "__main__":
    test_question = "What is the collateral free loan limit for MSME?"
    ask_ai(test_question)

    print("\n=======================================")
    ask_ai("What is the new collateral free limit based on the February 2026 amendment?")
    
    # Ask Question 2
    print("\n=======================================")
    ask_ai("What is the timeline for credit decisions for loans up to ₹25 lakh?")