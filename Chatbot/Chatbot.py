from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
from operator import itemgetter
import langchain

class Chatbot:
    langchain.debug = True
    def __init__(self, chat_history, user_info):
        global llm, retriever, vectorstore
        llm = LlamaCpp(
            model_path="./models/mistral-7b-instruct-v0.1.Q4_0.gguf",
            temperature=1.0,
            max_tokens=1000,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )
        self.chat_history = chat_history
        self.user_info = user_info
        save_directory = "faiss_index"
        vectorstore = FAISS.load_local(
            save_directory, 
            LlamaCppEmbeddings(
                model_path="models/mxbai-embed-large-v1-f16.gguf", 
                n_gpu_layers=-1,
                n_ctx=512,
                verbose = False,),
            allow_dangerous_deserialization=True)
            
    def query(self, question):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def format_chat_history(chat_history):
            count = 1
            formatted_chat_history = ""
            for item in chat_history:
                formatted_chat_history += f"Question {count}: {item['question']}\nAnswer {count}: {item['answer']}\n"
                count += 1
            return formatted_chat_history
        
        answer_prompt = PromptTemplate.from_template(
"""You are a personal health chatbot that provides guidance on health and wellness in a conversational manner.
You are asked a question by a user and you provide an answer, and nothing more.
DO NOT MAKE UP QUESTIONS.
Here is some information to help you answer the user's question:
{context}
Here is a list of the user's information:
Name: {name}, Age: {age}, Weight: {weight}, Height: {height}, Gender: {gender}
Here is your chat history with the user:
{chat_history}
Final Question: {question}
Final Answer:""")
        
        def retrieve_docs(question):
            return format_docs(vectorstore.similarity_search(question["question"]))
        
        answer_chain = (
            {
                "context": retrieve_docs,
                "question": itemgetter("question"), 
                "chat_history": itemgetter("chat_history"),
                "name": itemgetter("name"),
                "age": itemgetter("age"),
                "weight": itemgetter("weight"),
                "height": itemgetter("height"),
                "gender": itemgetter("gender"),
            }
            | answer_prompt 
            | llm
            )
        answer = answer_chain.invoke(
            {
                "question":question, 
                "chat_history": format_chat_history(self.chat_history), 
                "name":self.user_info["name"],
                "age":self.user_info["age"],
                "weight":self.user_info["weight"],
                "height":self.user_info["height"],
                "gender":self.user_info["gender"],
            })
        self.chat_history.append({"question":question, "answer":answer})
        return answer
    
    def get_history(self):
        return self.chat_history