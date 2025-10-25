from dotenv import load_dotenv
from langchain_promt import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes")
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.invoke(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x,to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, second=invoke_model, third=parse_output)

response = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(response)