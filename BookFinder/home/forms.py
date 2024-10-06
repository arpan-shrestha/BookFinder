from django import forms
from .models import Books

class BookForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ['title','author','pdf_path']