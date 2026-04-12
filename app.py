import os
from langchain_openai import ChatOpenAI

# Load API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(
    base_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo",
    temperature=0
)

response = llm.invoke("Hi")

print(response.content)