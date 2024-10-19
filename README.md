# BookFinder Search System

A Django-based document retrieval system that ranks books using a Vector Space Model with TF-IDF and cosine similarity.

## Features
- **Efficient Book Search**: Ranks books based on query relevance.
- **Text Preprocessing**: Includes lemmatization, tokenization, and stopword removal.
- **Multi-threading**: Optimizes text file processing for fast response.
- **Database Integration**: Displays book metadata (author name, cover) in search results.

## Technologies Used
- **Django**: Used as the backend framework to handle server-side logic and database integration.
- **NLTK (Natural Language Toolkit)**: Employed for natural language processing tasks, including tokenization, stopword removal, and lemmatization.
- **scikit-learn**: Utilized for implementing cosine similarity calculations to rank books based on relevance.
- **HTML/CSS**: For designing the user interface and creating a visually appealing search page.
