from django.http import JsonResponse
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from .preload import FAISSLoader
from rest_framework.response import Response
from rest_framework.decorators import api_view
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Global memory storage with limited history
MAX_MEMORY = 2  # Store the last 5 messages
persistent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_memory=MAX_MEMORY)

def index(request):
    """ Endpoint to check FAISS database availability """
    try:
        FAISSLoader.get_faiss_db()
        print("FAISS database accessed successfully!")
        return JsonResponse({"message": "FAISS database loaded successfully!"})
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=500)

@api_view(['POST'])
def chatbot_response(request):
    """
    API endpoint for handling chatbot responses.
    """
    prompt = request.data.get('prompt')

    if not prompt:
        return Response({"error": "No prompt provided"}, status=400)

    try:
        # Load FAISS database
        vectorstore = FAISSLoader.get_faiss_db()

        # Get conversation chain with persistent memory
        conversation = get_conversation_chain(vectorstore)

        # Generate a response
        response = conversation({"question": prompt})

        print(response)

        # Extract the answer from the response
        return Response({"response": response.get('answer', "I'm not sure how to answer that.")})
    
    except Exception as e:
        return Response({"error": str(e)}, status=500)

def get_conversation_chain(vectorstore):
    """
    Creates and returns a conversational retrieval chain with persistent memory.
    """
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=persistent_memory  # Global memory with limited history
    )

    print("\n4. Chain created with persistent memory (limited)\n")
    return conversation_chain
