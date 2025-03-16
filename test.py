import flask
import spacy
from transformers import pipeline

# Print library versions
print("Flask version:", flask.__version__)
print("SpaCy version:", spacy.__version__)

# Initialize summarization pipeline with a smaller model
summarizer = pipeline("summarization", model="t5-small")

# Test summarization
text = "This is a test sentence to check if the summarization pipeline works."
summary = summarizer(text, max_length=5, min_length=1)
print("Summary:", summary)