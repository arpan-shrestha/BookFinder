from django.contrib import admin
from .models import Book
import pdfplumber
# Register your models here.

class BookAdmin(admin.ModelAdmin):
    def save_model(self, request, obj, form, change):
        if obj.pdf_path:
            pdf_path = obj.pdf_path
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    text = ''.join([page.extract_text() for page in pdf.pages])
                obj.content = text
                obj.preprocess_contect()
            except FileNotFoundError:
                obj.content = "Error: PDF file not found at the specified path."
        super().save_model(request, obj, form, change)


admin.site.register(Book, BookAdmin)