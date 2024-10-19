import os
import re
import math
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import nltk
from .models import Book

# Make sure to download NLTK data files if you haven't already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nltk.data.path.append('/Users/arpanshrestha/Desktop/final/myenv/lib/nltk_data')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Path where text files are stored
txt_path = os.path.join(settings.BASE_DIR, 'corpus')

# Text preprocessing components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess the text (lowercase, remove punctuation, remove stopwords, and lemmatize)
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]  # Remove stopwords and lemmatize
    return tokens

# Term Frequency calculation
def term_frequency(term, document):
    term_count = document.count(term)
    total_terms = len(document)
    return term_count / total_terms if total_terms > 0 else 0 

# Inverse Document Frequency calculation
def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for document in all_documents if term in document)
    if num_docs_containing_term == 0:  # Avoid division by zero
        return 0
    return math.log(len(all_documents) / num_docs_containing_term)

# Function to read and preprocess a single document (for threading)
def read_and_preprocess_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    processed_text = preprocess(text)
    return processed_text


# View to handle the search query and return ranked results
def search_books(request):
    if request.method == "GET":
        books = Book.objects.all()
        return render(request, 'search.html', {'books':books})
    
    elif request.method == "POST":
        query = request.POST.get('q', '')  # Capture the search query
        results = []

        if query:
            processed_query = preprocess(query)

            # Multi-threaded document processing
            documents = []
            filenames = []

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_filename = {
                    executor.submit(read_and_preprocess_file, os.path.join(txt_path, filename)): filename
                    for filename in os.listdir(txt_path) if filename.endswith(".txt")
                }

                for future in future_to_filename:
                    filename = future_to_filename[future]
                    try:
                        processed_text = future.result()
                        documents.append(processed_text)
                        filenames.append(filename)
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")

            all_terms = set([term for doc in documents for term in doc]).union(set(processed_query))

            # Calculate TF-IDF for documents
            tfidf_documents = []
            for doc in documents:
                tfidf_vector = []
                for term in all_terms:
                    tf = term_frequency(term, doc)
                    idf = inverse_document_frequency(term, documents)
                    tfidf_vector.append(tf * idf)
                tfidf_documents.append(tfidf_vector)

            # Calculate TF-IDF for the query
            tfidf_query = []
            for term in all_terms:
                tf = term_frequency(term, processed_query)
                idf = inverse_document_frequency(term, documents)
                tfidf_query.append(tf * idf)

            tfidf_documents = np.array(tfidf_documents)
            tfidf_query = np.array([tfidf_query])

            # Calculate cosine similarities
            cosine_similarities = cosine_similarity(tfidf_query, tfidf_documents).flatten()
            ranked_results = sorted(zip(filenames, cosine_similarities), key=lambda x: x[1], reverse=True)

            # Retrieve author names and book covers from the database
            results = []
            for filename, similarity in ranked_results:
                try:
                    # Fetch book data from the database using the filename
                    book = Book.objects.get(filename=filename)
                    results.append({
                        'filename': filename,
                        'similarity': similarity,
                        'author_name': book.author_name,
                        'book_cover': book.book_cover.url if book.book_cover else None,
                    })
                except Book.DoesNotExist:
                    # Handle the case where the book is not found in the database
                    results.append({
                        'filename': filename,
                        'similarity': similarity,
                        'author_name': 'Unknown Author',
                        'book_cover': None,
                    })

        return render(request, 'search_results.html', {"query": query, "results": results})
