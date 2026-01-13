import io
from pypdf import PdfReader

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    ext = filename.split('.')[-1].lower()
    text = ""
    
    try:
        if ext == 'pdf':
            reader = PdfReader(io.BytesIO(file_content))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception:
        pass
        
    return text
