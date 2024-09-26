from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from perplexity import Perplexity

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

class PerplexityWebUIChatModel(BaseChatModel):
    """
    The email associated with the Perplexity account.
    """
    email: str
    backend_uuid: Optional[str] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        followup: bool = False,
        with_links: bool = False,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Call the Perplexity API.
        """
        perplexity = Perplexity(self.email)
        last_message = messages[-1]
        try:
            tokens = last_message.content
            resp = perplexity.search_sync(tokens, backend_uuid=self.backend_uuid if followup else None, timeout=None, **kwargs)
            self.backend_uuid = resp['backend_uuid']
            if 'error' in resp:
                return resp['error']
            body = resp['text']
            # Provide the answer to the user.
            text = body['answer']

            if with_links:
                # Provide the links to the user with minimal formatting.
                links = body['web_results']
                if len(links) > 0:
                    text += "\n\n"
                for i, link in enumerate(links):
                    text += f"[{i+1}] {link['url']}\n"
            self.backend_uuid = resp['backend_uuid']
        except Exception as e:
            raise e
        finally:
            perplexity.close()

        del resp['text']
        message = AIMessage(content=text, additional_kwargs={}, response_metadata=resp)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        followup: bool = False, 
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the Perplexity API.
        """
        perplexity = Perplexity(self.email)
        last_message = messages[-1]
        try:
            tokens = last_message.content
            streamed_resp = perplexity.search(tokens, backend_uuid=self.backend_uuid if followup else None, timeout=None, **kwargs)
            for resp in streamed_resp:
                self.backend_uuid = resp['backend_uuid']
                raw_chunks = resp['text']['chunks']
                if len(raw_chunks) == 0:
                    continue
                
                raw_chunk = raw_chunks[-1]
                is_last = resp['status'] == 'completed'
                del resp['text']
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=raw_chunk, response_metadata={
                    'backend_uuid': resp['backend_uuid'],
                } if is_last else {}))
                yield chunk

                if is_last or (stop is not None and raw_chunk in stop):
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
