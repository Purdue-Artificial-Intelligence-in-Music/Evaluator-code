from django.urls import path

from .views import UploadImageView
from .views import UploadVideoView

urlpatterns = [
    path('upload/', UploadImageView.as_view(), name='upload_image'),
    path('send-video/', UploadVideoView.as_view(), name='upload_video')

]