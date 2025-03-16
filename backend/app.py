# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from text_extractor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_image,
    clean_text,
    preprocess_text
)
from summarizer import summarize_text, extract_keywords, translate_text
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "Legal Document Summarizer Backend is running!"

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Step 1: Get the uploaded file
        file = request.files['file']
        file_type = file.filename.split('.')[-1].lower()

        # Step 2: Extract text based on file type
        if file_type == 'pdf':
            complete_text = extract_text_from_pdf(file)
        elif file_type == 'docx':
            complete_text = extract_text_from_docx(file)
        elif file_type in ['jpg', 'jpeg', 'png']:
            complete_text = extract_text_from_image(file)
        else:
            complete_text = file.read().decode('utf-8')

        # Step 3: Clean and preprocess the text
        cleaned_text = clean_text(complete_text)
        preprocessed_text = preprocess_text(cleaned_text)

        # Step 4: Summarize the text
        summary = summarize_text(preprocessed_text, max_summary_words=1000)

        # Step 5: Extract keywords
        keywords = extract_keywords(preprocessed_text, top_n=10)

        # Step 6: Translate the summary and keywords into Hindi
        translated_summary = translate_text(summary, target_language='hi')
        translated_keywords = [translate_text(keyword, target_language='hi') for keyword in keywords]

        # Step 7: Return the results
        return jsonify({
            'complete_text': complete_text,
            'summary': summary,
            'keywords': keywords,
            'translated_summary': translated_summary,
            'translated_keywords': translated_keywords
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)