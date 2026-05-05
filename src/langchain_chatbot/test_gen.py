from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # stable + fast
)

response = llm.invoke([
    HumanMessage(content="Explain what LangChain does")
])

print(response.content)