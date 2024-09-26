from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from perplexity import Perplexity


class PerplexityWebUILLM(LLM):
    """
    The email associated with the Perplexity account.
    """
    email: str
    backend_uuid: Optional[str] = None
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        followup: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Call the Perplexity API.
        """
        perplexity = Perplexity(self.email)
        try:
            resp = perplexity.search_sync(prompt, backend_uuid=self.backend_uuid if followup else None, timeout=None, **kwargs)
            self.backend_uuid = resp['backend_uuid']
            if 'error' in resp:
                return resp['error']
            body = resp['text']
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
        followup: bool = False, 
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream the Perplexity API.
        """
        perplexity = Perplexity(self.email)
        try:
            streamed_resp = perplexity.search(prompt, backend_uuid=self.backend_uuid if followup else None, timeout=None, **kwargs)
            for resp in streamed_resp:
                self.backend_uuid = resp['backend_uuid']
                chunks = resp['text']['chunks']
                if len(chunks) == 0:
                    continue
                
                chunk = chunks[-1]
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