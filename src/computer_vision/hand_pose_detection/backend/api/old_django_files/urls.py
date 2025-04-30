from django.urls import path

from .views import UploadImageView, UploadVideoView, FilePathVideo

urlpatterns = [
    path('upload/', UploadImageView.as_view(), name='upload_image'),
    path('send-video/', UploadVideoView.as_view() ,name = 'upload_video'),
    path('change-video/', FilePathVideo.as_view() ,name = 'change_video')
    ]