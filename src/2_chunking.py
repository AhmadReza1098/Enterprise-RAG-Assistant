import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/raw_documents/rbi_msme_2026.pdf"

# 1. The extraction function we already know works
def extract_text(path):
    print(f"Reading document: {path}...")
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:3]: # We will stick to 3 pages for this test
            text += page.extract_text() + "\n"
    return text

# 2. Our new chunking function
def chunk_text(text):
    print("Slicing text into AI-readable chunks...")
    
    # Set up the logic for how we want to chop the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # Max number of characters per chunk
        chunk_overlap=50    # Overlap by 50 characters to keep context between chunks
    )
    
    # Do the actual slicing
    chunks = splitter.split_text(text)
    return chunks

# Run the script
if __name__ == "__main__":
    raw_text = extract_text(PDF_PATH)
    text_chunks = chunk_text(raw_text)
    
    print(f"\nSuccess! Sliced the pages into {len(text_chunks)} distinct chunks.")
    print("\n--- Preview of Chunk #1 ---")
    print(text_chunks[0])
    
    print("\n--- Preview of Chunk #2 ---")
    print(text_chunks[1])
