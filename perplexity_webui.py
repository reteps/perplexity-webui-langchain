from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from perplexity import Perplexity
import json


class PerplexityWebUILLM(LLM):
    """
    The email associated with the Perplexity account.
    """
    email: str
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the Perplexity API.
        """
        perplexity = Perplexity(self.email)
        try:
            answer = perplexity.search_sync(prompt)
            if 'error' in answer:
                return answer['error']
            body = json.loads(answer['text'])
            # Provide the answer to the user.
            text = body['answer']
            # Provide the links to the user with minimal formatting.
            links = body['web_results']
            if len(links) > 0:
                text += "\n\n"
            for i, link in enumerate(links):
                text += f"[{i+1}] {link['url']}\n"
        except Exception as e:
            raise e
        finally:
            perplexity.close()
        return text
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream the Perplexity API.
        """
        perplexity = Perplexity(self.email)
        try:
            streamed_resp = perplexity.search(prompt)
            for resp in streamed_resp:
                if 'chunks' in resp:
                    if len(resp['chunks']) == 0:
                        continue
                    chunk = resp['chunks'][-1]
                else:
                    chunk = json.loads(resp['text'])['chunks'][-1]
                gen_chunk = GenerationChunk(text=chunk)
                if run_manager:
                    run_manager.on_llm_new_token(gen_chunk.text, chunk=gen_chunk)
                yield gen_chunk
                if resp['status'] == 'completed':
                    break
        except Exception as e:
            raise e
        finally:
            perplexity.close()

    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": "PerplexityWebUILLM"}

    @property
    def _llm_type(self) -> str:
        return "custom"