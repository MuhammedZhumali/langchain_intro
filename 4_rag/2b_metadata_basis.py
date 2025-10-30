import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persist_directory = os.path.join(db_dir, "chroma_metadata_basis")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

query = "What is a robot?"

retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.1})
relevant_docs = retriever.invoke(query)


print("Relevant documents:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:500]}...")