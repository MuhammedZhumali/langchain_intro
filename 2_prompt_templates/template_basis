from langchain.prompt import ChatPromptTemplate
from langchain.schema import HumanMessage

template = "Tell me a joke about {topic}."

prompt_template = ChatPromptTemplate.fromt_template(template)

print("Prompt from template")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)
