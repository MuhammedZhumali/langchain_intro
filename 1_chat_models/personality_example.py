from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

model = ChatOpenAI(model_name="gpt-4")

# Example 1: Math Teacher
math_messages = [
    SystemMessage(content="You are a math teacher. Explain concepts in a clear, educational way."),
    HumanMessage(content="What is a prime number?")
]

# Example 2: Pirate
pirate_messages = [
    SystemMessage(content="You are a friendly pirate. Speak in pirate style while explaining things."),
    HumanMessage(content="What is a prime number?")
]

# Example 3: Poet
poet_messages = [
    SystemMessage(content="You are a romantic poet. Explain concepts through beautiful metaphors and verse."),
    HumanMessage(content="What is a prime number?")
]

print("=== Math Teacher Response ===")
result = model.invoke(math_messages)
print(result.content)

print("\n=== Pirate Response ===")
result = model.invoke(pirate_messages)
print(result.content)

print("\n=== Poet Response ===")
result = model.invoke(poet_messages)
print(result.content)