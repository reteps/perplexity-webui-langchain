from perplexity_webui_langchain import PerplexityWebUIChatModel
from langchain_core.messages import HumanMessage

import asyncio

llm = PerplexityWebUIChatModel(email="peteras4@illinois.edu")

async def main():
    msg = await llm.ainvoke([
        HumanMessage(content="What is the capital of Illinois?"),
    ])
    print(msg.content)
    async for msg in llm.astream("hello"):
        print(msg.content, end="|", flush=True)

    async for msg in llm.astream("what was my last message?", followup=True):
        print(msg.content, end="|", flush=True)
if __name__ == "__main__":
    asyncio.run(main())