import re
import string
import io
from typing import Union
from pypdf import PdfReader
import docx

def clean_resume(text: str) -> str:
    """
    Cleans resume text by removing URLs, RT/cc flags, hashtags,
    mentions, punctuations, numbers, and extra whitespaces.
    Matches the original preprocessing pipeline.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    # Remove RT and cc
    text = re.sub(r'\b(RT|cc)\b', ' ', text)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    # Remove numbers
    text = re.sub(r'[0-9]+', ' ', text)
    # Remove non-ascii characters
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    # Remove redundant whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_pdf(file_source: Union[str, io.BytesIO, bytes]) -> str:
    """Extract text from a PDF file path or byte buffer."""
    if isinstance(file_source, bytes):
        file_source = io.BytesIO(file_source)
    reader = PdfReader(file_source)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(file_source: Union[str, io.BytesIO, bytes]) -> str:
    """Extract text from a Word DOCX file path or byte buffer."""
    if isinstance(file_source, bytes):
        file_source = io.BytesIO(file_source)
    doc = docx.Document(file_source)
    text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text]
    return "\n".join(text)


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """Extract text based on file extension from Streamlit UploadedFile or file-like object."""
    file_name = getattr(uploaded_file, "name", "").lower()
    content = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(content)
    elif file_name.endswith(".txt"):
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)
    else:
        # Fallback to UTF-8 decoding
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)
