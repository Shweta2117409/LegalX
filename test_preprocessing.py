# test_preprocessing.py
from backend.text_extractor import extract_text_from_file, clean_text, preprocess_text
from backend.summarizer import summarize_text, extract_keywords, translate_text

def test_text_extraction_and_summarization():
    # Replace this path with the actual path to your sample PDF file
    pdf_path = "PROBLEM STATEMENT 2.pdf"

    # Test PDF extraction
    try:
        # Step 1: Extract text from the PDF
        pdf_text = extract_text_from_file(pdf_path)
        print("Extracted Text (First 1000 characters):\n", pdf_text[:1000])

        # Step 2: Clean the text
        cleaned_text = clean_text(pdf_text)
        print("\nCleaned Text (First 1000 characters):\n", cleaned_text[:1000])

        # Step 3: Preprocess the text
        preprocessed_text = preprocess_text(cleaned_text)
        print("\nPreprocessed Text (First 1000 characters):\n", preprocessed_text[:1000])

        # Step 4: Extract keywords
        keywords = extract_keywords(preprocessed_text, top_n=10)
        print("\nTop 10 Keywords:\n", keywords)

        # Step 5: Summarize the text (within 1000 words)
        summary = summarize_text(preprocessed_text, max_summary_words=1000)
        print("\nComplete Summary (Within 1000 words):\n", summary)

        # Step 6: Translate the summary and keywords into Hindi
        translated_summary = translate_text(summary, target_language='hi')
        translated_keywords = [translate_text(keyword, target_language='hi') for keyword in keywords]

        print("\nTranslated Summary (Hindi):\n", translated_summary)
        print("\nTranslated Keywords (Hindi):\n", translated_keywords)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_text_extraction_and_summarization()