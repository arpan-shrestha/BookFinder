import os
import re
import math
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Make sure to download NLTK data files if you haven't already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nltk.data.path.append('/Users/arpanshrestha/Desktop/final/myenv/lib/nltk_data')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Path where text files are stored
txt_path = os.path.join(settings.BASE_DIR, 'pdfs')

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


# View to handle the search query and return ranked results
def search_books(request):
    if request.method == "GET":
        # Show the search page with an empty form when the page is loaded
        return render(request, 'search.html')
    
    elif request.method == "POST":
        # Process the query once submitted
        query = request.POST.get('q', '')  # Capture the search query
        results = []

        if query:
            processed_query = preprocess(query)

            # Process all documents (text files) in the directory
            documents = []
            filenames = []
            for filename in os.listdir(txt_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(txt_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    processed_text = preprocess(text)
                    documents.append(processed_text)
                    filenames.append(filename)

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

            results = ranked_results

        # After processing the search query, show the results in search_results.html
        return render(request, 'search_results.html', {"query": query, "results": results})


