# text_extractor.py
from PyPDF2 import PdfReader  # Use PyPDF2 for PDF text extraction
from docx import Document
import pytesseract
from PIL import Image
import os
import re
import spacy

# Load the SpaCy model for English
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyPDF2.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    """
    Extracts text from a DOCX file.

    Args:
        docx_path (str): Path to the DOCX file.

    Returns:
        str: Extracted text from the DOCX file.
    """
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(image_path):
    """
    Extracts text from an image file using OCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text from the image.
    """
    return pytesseract.image_to_string(Image.open(image_path))

def extract_text_from_file(file_path):
    """
    Extracts text from a file (PDF, DOCX, or image).

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Extracted text from the file.
    """
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    # Extract text based on file type
    if file_extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension == ".docx":
        return extract_text_from_docx(file_path)
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def clean_text(text):
    """
    Cleans the text by removing extra spaces, special characters, and irrelevant sections.

    Args:
        text (str): Raw text extracted from the document.

    Returns:
        str: Cleaned text.
    """
    # Remove extra spaces and special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()  # Remove leading/trailing spaces

def preprocess_text(text):
    """
    Preprocesses the text by tokenizing, removing stopwords, and punctuation.

    Args:
        text (str): Cleaned text.

    Returns:
        str: Preprocessed text.
    """
    doc = nlp(text)
    # Remove stopwords and punctuation, and keep only alphabetic tokens
    cleaned_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(cleaned_tokens)