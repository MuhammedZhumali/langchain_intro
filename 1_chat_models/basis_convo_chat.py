from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = ChatOpenAI(model_name="gpt-4o")

message = [
    SystemMessage(content="Solve math problems."),
    HumanMessage(content="What is 2+2?")
]

result = model.invoke(message)
print(f"Answer from AI: {result.content}")


messages = [
    SystemMessage(content="Solve math problems."),
    HumanMessage(content="What is 2+2?"),
    AIMessage(content="2+2 is 4."),
    HumanMessage(content="What is 39999*12?")
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")