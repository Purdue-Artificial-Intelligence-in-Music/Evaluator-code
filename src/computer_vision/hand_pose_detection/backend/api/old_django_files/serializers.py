from rest_framework import serializers
from .models import UploadedImage, Article
from drf_extra_fields.fields import Base64ImageField

class UploadedImageSerializer(serializers.ModelSerializer):
    image = Base64ImageField(required=False)
    class Meta:
        model = Article
        fields = ("title", "content", "image")