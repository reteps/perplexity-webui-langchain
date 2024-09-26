# from .perplexity_webui import PerplexityWebUILLM
import asyncio
import perplexity_webui

llm = perplexity_webui.PerplexityWebUILLM(email="peteras4@illinois.edu")

# res = llm.invoke("What is the capital of Illinois?")
# print(res)
async def main():
    # res = await llm.ainvoke("What is the capital of Illinois?")
    # print(res)
    async for token in llm.astream("hello"):
        print(token, end="|", flush=True)

    async for token in llm.astream("what was my last message?", followup=True):
        print(token, end="|", flush=True)
if __name__ == "__main__":
    asyncio.run(main())