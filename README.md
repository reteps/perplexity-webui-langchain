# perplexity-webui-langchain

![](./perplexity-webui-logo.png)

A wrapper around the Perplexity WebUI using Langchain

## Motivation

- The perplexity pro plan supports custom LLMs (e.g. claude3.5) on the WebUI (and can be scripted through a WebSocket)
- I want to be able to use these custom LLMs in other settings, like autocomplete in my code editor
- I can hook up Langchain as an OpenAI chat completions compatible endpoint using [this repo](https://github.com/samuelint/langchain-openai-api-bridge)

![](./perplexity-webui-langchain.png)