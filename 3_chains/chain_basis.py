from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.prompt import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes")
    ]
)

chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model (no content parsing)

result = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(result)