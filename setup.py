from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="perplexity_webui_langchain",
    version="0.1",
    description="A python api to use the perplexity.ai webui in langchain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reteps/perplexity-webui-langchain",
    packages=find_packages(),
    install_requires=[
        "perplexityai@git+https://github.com/reteps/perplexityai.git",
        "langchain",
    ],
)
