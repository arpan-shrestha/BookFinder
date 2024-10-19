from django.db import models

class Book(models.Model):
    filename = models.CharField(max_length=255, unique=True)
    author_name = models.CharField(max_length=255)
    book_cover = models.ImageField(upload_to='book_covers/')

    def __str__(self):
        return self.filename