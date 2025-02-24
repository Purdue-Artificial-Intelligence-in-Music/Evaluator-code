from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import UploadedImageSerializer

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from .backend import processFrame

from django.http import JsonResponse

@method_decorator(csrf_exempt, name='dispatch')
class UploadImageView (APIView): 
    @csrf_exempt
    def post(self, request):
        print("received") # check the request data

        serializer = UploadedImageSerializer(data=request.data)
        if serializer.is_valid():
            article = serializer.save()
            return Response(serializer.data, status=201)
        else:
            return Response(serializer.errors, status=400)

'''
def hello(request):
    data = {
        'name': 'image',
        'type': 'image/png',
    }
    
    uploaded_file = request.FILES['file']
    file_name = default_storage.save(uploaded_file.name, uploaded_file) # Save to default storage
    return JsonResponse({'message': 'File uploaded successfully', 'file_url': default_storage.url(file_name)})
    '''