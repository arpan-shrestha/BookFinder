from django.db import models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
# Create your models here.

class Book(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    pdf_file = models.CharField(max_length=500)


    def preprocess_content(self):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        text = re.sub(r'[^\w\s]', '', self.content)
        text = re.sub(r'\d+','',text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        self.tokens = tokens
        self.save()

    def __str__(self):
        return self.title