from django.shortcuts import render

from django.http import JsonResponse

def hello(request):

    data = {"message": "hi from django"}
    response = JsonResponse(data)

    # Manually add CORS header
    response["Access-Control-Allow-Origin"] = "*" 
    return response
