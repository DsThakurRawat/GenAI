import pypdf
import sys

def search_pdf(pdf_path, keyword):
    try:
        reader = pypdf.PdfReader(pdf_path)
        matches = []
        for i in range(len(reader.pages)):
            text = reader.pages[i].extract_text()
            if keyword.lower() in text.lower():
                matches.append(i + 1)
        if matches:
            print(f"Keyword '{keyword}' found on pages: {matches}")
        else:
            print(f"Keyword '{keyword}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_pdf("LANGCHAIN AND genAI notes.pdf", "Prompt")
