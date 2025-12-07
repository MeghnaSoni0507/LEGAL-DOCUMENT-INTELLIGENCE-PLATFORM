import re

def clean_ocr_text(text):
    """
    Clean OCR-extracted text by removing excessive whitespace,
    fixing common OCR errors, and standardizing formatting.
    """
    if not text:
        return ""
    
    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces (more than 1)
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix common OCR errors
    text = text.replace('|', 'I')  # Vertical bars often misread
    text = text.replace('0', 'O').replace('O', '0')  # Context-dependent
    
    # Remove page markers if present
    text = re.sub(r'\[Page \d+\]\s*', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text