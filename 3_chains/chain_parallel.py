from dotenv import load_dotenv
from langchain_prompt import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human", "List the main features of the product: {product_name}"),
    ]
)

def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given the following features, list pros:\n{features}"),
        ]
    )
    return pros_template.format_prompt(features=features)

def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given the following features, list cons:\n{features}"),
        ]
    )
    return cons_template.format_prompt(features=features)

def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\nCons:\n{cons}"

pros_branch = (
    RunnableLambda(lambda x: analyze_pros(x) | model | StrOutputParser())
)

cons_branch = (
    RunnableLambda(lambda x: analyze_cons(x) | model | StrOutputParser())
)

chain = (
    prompt_template | model | StrOutputParser() | 
    RunnableParallel(branches={"pros": pros_branch, "cons": cons_branch}) |
    RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name": "MacBook Pro M4"})

print(result)