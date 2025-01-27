from dotenv import load_dotenv
import firebase_admin
import os
from firebase_admin import credentials, firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_mistralai import ChatMistralAI

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

model = ChatMistralAI(api_key=api_key)

cred = credentials.Certificate("./config/langbase.json")
app = firebase_admin.initialize_app(cred)

client = firestore.client(app)

chat_history = FirestoreChatMessageHistory(
    session_id="current_session", 
    collection="chat_history", 
    client=client
)

while True:
    human_input = input("You: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)
    ai_response = model.invoke(chat_history.messages)

    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
