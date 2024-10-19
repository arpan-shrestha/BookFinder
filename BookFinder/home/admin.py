from django.contrib import admin
from .models import Book

class BookAdmin(admin.ModelAdmin):
    list_display = ('author_name', 'filename') 

admin.site.register(Book, BookAdmin)