import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FOLDER_PATH = "data/raw_documents/"
DB_PATH = "data/vector_db"

# 1. Our brand new function to read multiple PDFs!
def extract_text_from_folder(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            
            # This print statement lets you watch it read each file in real-time!
            print(f"📄 Reading: {filename}...") 
            
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n"
    return all_text

if __name__ == "__main__":
    print("Starting the multi-document extraction...")
    
    # 2. Extract text from ALL PDFs in the folder
    raw_text = extract_text_from_folder(FOLDER_PATH)
    
    if raw_text.strip() == "":
        print("⚠️ No text found! Make sure you have PDFs in the folder.")
    else:
        # 3. Slice the massive text wall into chunks
        print("\n✂️ Slicing text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        print(f"Created {len(chunks)} chunks.")
        
        # 4. Convert chunks to math and save to the FAISS database
        print("\n🧠 Building the Vector Database...")
        embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_texts(chunks, embeddings_model)
        
        vector_db.save_local(DB_PATH)
        print("\n✅ Database successfully updated with all documents!")