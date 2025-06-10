from django.urls import path
from . import views

urlpatterns = [
    path('hello/', views.index, name='chatbotresponse'),
    path('chatbotresponse', views.chatbot_response, name='mainresponse'),
]