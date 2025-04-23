from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=255, null=False, blank=False)
    content = models.TextField(max_length=5000, null=False, blank=False)
    image = models.ImageField(null=False, blank=True) # upload to parameter eliminated
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="created_at")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="updated_at")
    def __str__(self):
        return self.title

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')