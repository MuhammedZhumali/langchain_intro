from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

result = model.invoke("HI, GPT, HOW ARE YOU?")
print("--- Result ---")
print(result)
print("--- Content ---")
print(result.content)