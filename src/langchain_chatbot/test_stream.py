# test_langchain_streaming_gemini.py
# pip install langchain langchain-google-genai

import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set your key: export GEMINI_API_KEY=your_key
# or pass api_key="..." directly below

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or gemini-2.0-flash for lighter/faster
    # api_key="your_key_here",
)

# --- 1. Basic token streaming ---
print("=== 1. Basic .stream() ===")
for chunk in model.stream([HumanMessage(content="Tell me a fun fact about Vietnam in 2 sentences.")]):
    print(chunk.content, end="", flush=True)
print("\n")

# --- 2. System + Human messages ---
print("=== 2. System + Human message streaming ===")
messages = [
    SystemMessage(content="You are a concise assistant. Answer in 2 sentences max."),
    HumanMessage(content="What is LangChain?"),
]
for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
print("\n")

# --- 3. LCEL chain streaming ---
print("=== 3. Chain streaming (prompt | model | parser) ===")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])
chain = prompt | model | StrOutputParser()

for chunk in chain.stream({"question": "List 3 benefits of streaming in LLMs."}):
    print(chunk, end="", flush=True)
print("\n")

# --- 4. Async streaming ---
print("=== 4. Async .astream() ===")
async def async_stream():
    async for chunk in model.astream([HumanMessage(content="What is Gemini 2.5 Flash?")]):
        print(chunk.content, end="", flush=True)
    print("\n")

asyncio.run(async_stream())

# --- 5. Stream with token usage ---
print("=== 5. Stream + token usage (last chunk) ===")
last_chunk = None
for chunk in model.stream([HumanMessage(content="Say hello in 5 languages.")]):
    print(chunk.content, end="", flush=True)
    last_chunk = chunk
print("\n")
if last_chunk and last_chunk.usage_metadata:
    print(f"Tokens used → input: {last_chunk.usage_metadata['input_tokens']}, "
          f"output: {last_chunk.usage_metadata['output_tokens']}")