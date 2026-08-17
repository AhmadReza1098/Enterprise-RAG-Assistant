from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "data/vector_db"

def test_search(query):
    print(f"Loading database from {DB_PATH}...")
    
    # We have to load the exact same AI model we used to build the database
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load our saved vault
    vector_db = FAISS.load_local(DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
    
    print(f"\nSearching for: '{query}'")
    
    # Search the database for the top 2 most relevant chunks
    results = vector_db.similarity_search(query, k=2)
    
    print("\n--- Top Results Found ---")
    for i, doc in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(doc.page_content)

if __name__ == "__main__":
    # We are asking a specific question about the RBI document
    test_query = "What is the collateral free loan limit for MSME?"
    test_search(test_query)