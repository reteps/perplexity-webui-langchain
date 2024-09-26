from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from perplexity import Perplexity

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessageChunk, AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages import HumanMessage, SystemMessage

print("loaded2")

def aider_hash_filter(x):
    # Ignore the hash of any AI message.
    if type(x) is AIMessage:
        print('ignoring', x)
        return False

    # Ignore the hash of any message that contains the string 'I am not sharing any files that you can edit yet.'
    if type(x.content) is list:
        for token in x.content:
            if token["type"] != "image_url":
                text = token["text"]
                break
    else:
        text = x.content
    
    if 'I am not sharing any files that you can edit yet.' in text:
        print('ignoring', x)
        return False

    return True

class PerplexityWebUIChatModel(BaseChatModel):
    # The email associated with the Perplexity account.
    email: str
    backend_uuid: Optional[str] = None

    '''
    The chat model has built-in support for continuing conversations.
    '''
    conversation_lookup: Dict[str, Tuple[str, List[str]]] = {}

    def _generate(
        self,
        input_: LanguageModelInput,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        followup: bool = False,
        with_links: bool = False,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Call the Perplexity API.
        """
        messages = self._convert_input(input_).to_messages()
        print("start_generate")
        print(messages)
        perplexity = Perplexity(self.email)
        last_message = messages[-1]
        try:
            tokens = last_message.content
            print("tokens")
            resp = perplexity.search_sync(
                tokens,
                backend_uuid=self.backend_uuid if followup else None,
                timeout=None,
                **kwargs,
            )
            self.backend_uuid = resp["backend_uuid"]
            if "error" in resp:
                return resp["error"]
            body = resp["text"]
            # Provide the answer to the user.
            text = body["answer"]

            if with_links:
                # Provide the links to the user with minimal formatting.
                links = body["web_results"]
                if len(links) > 0:
                    text += "\n\n"
                for i, link in enumerate(links):
                    text += f"[{i+1}] {link['url']}\n"
            self.backend_uuid = resp["backend_uuid"]
        except Exception as e:
            raise e
        finally:
            perplexity.close()

        del resp["text"]
        message = AIMessage(content=text, additional_kwargs={}, response_metadata=resp)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        input_: LanguageModelInput,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        followup: bool = False,
        hash_filter: Any = aider_hash_filter,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the Perplexity API.


        followup lets you continue the conversation from the last message.
        By default, followup works for multiple messages from 1 call.

        
        """
        messages = self._convert_input(input_).to_messages()
        perplexity = Perplexity(self.email)
        backend_uuid = self.backend_uuid
        print([str(type(m)) for m in messages])
        try:
            # Before we send everything to Perplexity, check if we can recover a partial message history from the conversation_lookup.
            # If we can, we can continue the conversation from the last message.

            attachments = []
            # Hash each message content to check if it is in the conversation_lookup.
            # If it is, we can continue the conversation from the last message.
            # If not, we can start a new conversation.

            # As a hack
            print(f'{self.conversation_lookup=}')
            hashable_messages = [m for m in messages if hash_filter(m)]
            j = len(hashable_messages)
            print(f'{hashable_messages=}')
            while j > 0:
                m_hash = str(hash(str(hashable_messages[:j])))
                if m_hash in self.conversation_lookup:
                    print(f'<< Found conversation in conversation_lookup: {m_hash}')
                    backend_uuid, attachments = self.conversation_lookup[m_hash]
                    messages = hashable_messages[j:]
                    followup = True
                    break
                else:
                    print(f'<< Did not find conversation[:{j}] in conversation_lookup: {m_hash}')
                j -= 1
            
            # Only add new messages to the conversation.
            num_messages = len(messages)
            for i, message in enumerate(messages):
                attachments_to_upload = []

                if type(message) is AIMessage:
                    print('(AIMessage)>> [discarded]')
                    continue

                tokens = message.content
                text_tokens = 0
                if type(tokens) is list:
                    for token in tokens:
                        if token["type"] == "image_url":
                            attachments_to_upload.append(token["image_url"]["url"])
                        else:
                            if text_tokens > 0:
                                raise ValueError(
                                    "Only one text token is allowed per message."
                                )
                            text_tokens += 1
                            text = token["text"]
                else:
                    text = tokens

                print(f'({type(message).__name__})>> {text}')
                # Upload the (new) attachments.
                attachments += [
                    perplexity.upload(attachment) for attachment in attachments_to_upload
                ]

                streamed_resp = perplexity.search(
                    text,
                    attachments=attachments,
                    backend_uuid=backend_uuid if followup else None,
                    timeout=None,
                    **kwargs,
                )
                first_chunk = True
                for resp in streamed_resp:
                    backend_uuid = resp["backend_uuid"]
                    raw_chunks = resp["text"]["chunks"]
                    if len(raw_chunks) == 0:
                        continue

                    raw_chunk = raw_chunks[-1]
                    is_last = resp["status"] == "completed"
                    del resp["text"]
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=raw_chunk,
                            response_metadata={
                                "backend_uuid": resp["backend_uuid"],
                            }
                            if is_last
                            else {},
                        )
                    )

                    if i == num_messages - 1:
                        yield chunk
                    
                    if first_chunk:
                        if i == num_messages - 1:
                            print('<< Sending response, as it is the last message in the chain.')
                        else:
                            print(f'<< Discarding response, as it is not the last message in the chain.')
                        first_chunk = False

                    if is_last or (stop is not None and raw_chunk in stop):
                        break
                
                # Set followup=True for the next message -- since it is a multi-message chain.
                followup = True
            # Hash the conversation to store it in the conversation_lookup.
            m_hash = str(hash(str(hashable_messages)))
            self.conversation_lookup[m_hash] = (backend_uuid, attachments)
            print(f'l[{m_hash}] = ({backend_uuid}, {attachments})')
            self.backend_uuid = backend_uuid
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
