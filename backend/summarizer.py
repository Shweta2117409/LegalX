# summarizer.py
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
import re

# Load the summarization pipeline with the t5-small model
summarizer = pipeline("summarization", model="t5-small")

def split_text_into_chunks(text, max_chunk_size=512):
    """
    Splits the text into smaller chunks of a specified size.

    Args:
        text (str): The input text.
        max_chunk_size (int): Maximum size of each chunk (default: 512).

    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def refine_summary(summary):
    """
    Refines the summary to improve punctuation, grammar, and readability.

    Args:
        summary (str): The raw summary text.

    Returns:
        str: The refined summary text.
    """
    # Add proper punctuation at the end of sentences
    summary = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+', '\n', summary)  # Add newlines after sentences
    summary = re.sub(r'\s+([.,!?])', r'\1', summary)  # Remove spaces before punctuation
    summary = re.sub(r'([.,!?])(?!\s|$)', r'\1 ', summary)  # Add spaces after punctuation

    # Capitalize the first letter of each sentence
    sentences = summary.split('. ')
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    summary = '. '.join(sentences)

    return summary

def summarize_text(text, max_length=130, min_length=30, max_summary_words=1000):
    """
    Summarizes the input text using the t5-small model, ensuring the final summary does not exceed max_summary_words.

    Args:
        text (str): The input text.
        max_length (int): Maximum length of each chunk's summary.
        min_length (int): Minimum length of each chunk's summary.
        max_summary_words (int): Maximum number of words in the final summary (default: 1000).

    Returns:
        str: The summarized text.
    """
    # Split the text into chunks
    chunks = split_text_into_chunks(text)

    # Summarize each chunk
    summaries = []
    total_words = 0
    for chunk in chunks:
        if total_words >= max_summary_words:
            break  # Stop if the summary has reached the maximum word limit
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summary_text = summary[0]['summary_text']
        summaries.append(summary_text)
        total_words += len(summary_text.split())

    # Combine the summaries into a final summary
    final_summary = " ".join(summaries)

    # Truncate the final summary to max_summary_words if necessary
    final_summary_words = final_summary.split()
    if len(final_summary_words) > max_summary_words:
        final_summary = " ".join(final_summary_words[:max_summary_words])

    # Refine the summary for proper punctuation and grammar
    final_summary = refine_summary(final_summary)

    return final_summary

def extract_keywords(text, top_n=10):
    """
    Extracts the top N keywords from the text using TF-IDF.

    Args:
        text (str): The input text.
        top_n (int): Number of top keywords to extract (default: 10).

    Returns:
        list: List of top keywords.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    sorted_indices = tfidf_scores.argsort()[::-1]
    return [feature_names[i] for i in sorted_indices[:top_n]]

def translate_text(text, target_language='hi'):
    """
    Translates the input text into the target language.

    Args:
        text (str): The input text.
        target_language (str): Target language code (e.g., 'hi' for Hindi).

    Returns:
        str: Translated text.
    """
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text