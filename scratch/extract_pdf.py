import pypdf
import sys

def extract_text(pdf_path, start_page=0, num_pages=20):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        end_page = min(start_page + num_pages, len(reader.pages))
        for i in range(start_page, end_page):
            text += f"--- Page {i+1} ---\n"
            text += reader.pages[i].extract_text() + "\n"
        print(text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_text("LANGCHAIN AND genAI notes.pdf", start_page=30, num_pages=10)
