from perplexity_webui_langchain import PerplexityWebUIChatModel
from langchain_core.messages import HumanMessage

import asyncio
import base64

llm = PerplexityWebUIChatModel(email="peteras4@illinois.edu")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def main():
    # msg = await llm.ainvoke([
    #     HumanMessage(content="What is the capital of Illinois?"),
    # ])
    # print(msg.content)
    # async for msg in llm.astream("hello"):
    #     print(msg.content, end="|", flush=True)

    # async for msg in llm.astream("what was my last message?", followup=True):
    #     print(msg.content, end="|", flush=True)

    encoded_image = encode_image("perplexity-webui-logo.png")
    print(len(encoded_image), encoded_image[:10])
    thread = llm.astream(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "What does the text in this image say"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ]
            )
        ]
    )

    async for msg in thread:
        print(msg.content, end="|", flush=True)


# stream = client.chat.completions.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": "what is machine learning?"}],
#     stream=True,
# )


if __name__ == "__main__":
    asyncio.run(main())
