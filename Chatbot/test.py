from Chatbot import Chatbot
chat_history = []
user_info = {
    "name": "Jane Doe",
    "age": "30",
    "weight": "120 lbs",
    "height": "5'5'",
    "gender": "Female",
}

chatbot = Chatbot(chat_history, user_info)

while (True):
    question = input("Ask a question, type \"Stop\" to stop:\n")
    if question.lower() == "stop":
        break
    print(chatbot.query(question))