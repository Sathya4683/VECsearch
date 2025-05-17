import os
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def generate_response(user_input,context):

    prompt_template = PromptTemplate(
        input_variables=["context", "input"],
        template="""
        You are a RAG (Retreival Augmented Generative) chatbot designed to answer the user query along with the context provided.
        Context: {context}
        User: {input}
        Assistant:
        """
    )
    prompt = prompt_template.format(context=context, input=user_input)
    response = llm.invoke(prompt)
    return response.content


#testing the function ... to check if the prompt_template works

# if __name__=="__main__":
#     user_inp = input("enter your query to ask the chatbot: ")
#     context = "user likes icecream"
#     output = generate_response(user_inp, context)
#     print(output)