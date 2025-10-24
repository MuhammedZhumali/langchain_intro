from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema improt AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

chat_message = []

system_message = SystemMessage(content="You are a helpful AI assistant")
chat_message.append(system_message)

while True:
    query = input("User message: ")
    if query.lower == "exit":
        break
    chat_message.append(HumanMessage(content=query))    

    result = model.invoke(chat_message)
    response = result.content
    chat_message.append(AIMessage(content=response))

    print(f"AI response: {response}")

print("Message history:")
print(chat_message)