from django.apps import AppConfig
from .preload import FAISSLoader

class Chatbot(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Chatbot'

    def ready(self):
        try:
            print("Calling preload_faiss...")
            FAISSLoader.preload_faiss()
            print("Preload completed successfully!")
        except Exception as e:
            print(f"Error during FAISS preload: {e}")