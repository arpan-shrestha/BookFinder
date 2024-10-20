# Generated by Django 5.1.1 on 2024-10-18 07:24

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('home', '0003_delete_book'),
    ]

    operations = [
        migrations.CreateModel(
            name='Book',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=255, unique=True)),
                ('author_name', models.CharField(max_length=255)),
                ('book_cover', models.ImageField(upload_to='book_covers/')),
            ],
        ),
    ]
