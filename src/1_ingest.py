import pdfplumber

# Define the exact path to your PDF
PDF_PATH = "data/raw_documents/rbi_msme_2026.pdf"

def extract_text_from_pdf(pdf_path):
    print(f"Opening document: {pdf_path}...")
    full_text = ""
    
    # Open the PDF using the tool we just installed
    with pdfplumber.open(pdf_path) as pdf:
        # Loop through just the first 3 pages to test it out
        for i, page in enumerate(pdf.pages[:3]):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
                print(f"Successfully read Page {i + 1}")
                
    return full_text

# Run the function
if __name__ == "__main__":
    extracted_text = extract_text_from_pdf(PDF_PATH)
    print("\n--- Preview of Extracted Text ---")
    print(extracted_text[:500]) # Print the first 500 characters
