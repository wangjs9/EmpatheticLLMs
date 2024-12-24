import os
import re
import json
import uuid
from abc import abstractmethod
from typing import Dict, List, Type
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_never
from tenacity import RetryError

from utils.message_utils import Message
from utils.config_utils import Config, BackendConfig, Configurable

try:
    import openai
    from openai import AzureOpenAI
    from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, \
        get_bearer_token_provider
    
    configs = json.load(open('utils/config.json', 'r', encoding='utf-8'))
    api_version = configs['azure_config']['api_version']
    if os.environ.get("END_POINT", None) is not None:
        os.environ["END_POINT"] = configs['azure_config']['endpoint']
    token_provider_link = configs['azure_config']['token_provider_link']
    azure_credential = ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(
            exclude_cli_credential=True,
            exclude_environment_credential=True,
            exclude_shared_token_cache_credential=True,
            exclude_developer_cli_credential=True,
            exclude_powershell_credential=True,
            exclude_interactive_browser_credential=True,
            exclude_visual_studio_code_credentials=True,
            managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
        )
    )
    token_provider = get_bearer_token_provider(azure_credential, token_provider_link)
    is_openai_available = True

except ImportError or openai.OpenAIError:
    is_openai_available = False

DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 1024
END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP = ("<|endoftext|>", END_OF_MESSAGE)  # End of sentence token
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"
SYSTEM_NAME = "System"


class IntelligenceBackend(Configurable):
    """An abstraction of the intelligence source of the agents."""
    
    stateful = None
    type_name = None
    
    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # registers the arguments with Configurable
    
    def __init_subclass__(cls, **kwargs):
        # check if the subclass has the required attributes
        for required in (
                "stateful",
                "type_name",
        ):
            if getattr(cls, required) is None:
                raise TypeError(
                    f"Can't instantiate abstract class {cls.__name__} without {required} attribute defined"
                )
        return super().__init_subclass__(**kwargs)
    
    def to_config(self) -> BackendConfig:
        self._config_dict["backend_type"] = self.type_name
        return BackendConfig(**self._config_dict)
    
    @abstractmethod
    def query(
            self,
            agent_name: str,
            role_desc: str,
            history_messages: List[Message],
            global_prompt: str = None,
            request_msg: Message = None,
            *args,
            **kwargs,
    ) -> str:
        raise NotImplementedError
    
    @abstractmethod
    async def async_query(
            self,
            agent_name: str,
            role_desc: str,
            history_messages: List[Message],
            global_prompt: str = None,
            request_msg: Message = None,
            *args,
            **kwargs,
    ) -> str:
        """Async querying."""
        raise NotImplementedError
    
    # reset the state of the backend
    def reset(self):
        if self.stateful:
            raise NotImplementedError
        else:
            pass


class OpenAIChatBot(IntelligenceBackend):
    stateful = False
    type_name = "azure-api"
    DEFAULT_MODEL = "gpt-4o"
    
    def __init__(
            self, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
            model: str = DEFAULT_MODEL, **kwargs
    ):
        """
        Instantiate the OpenAIChat backend.

        args:
            temperature: the temperature of the sampling
            max_tokens: the maximum number of tokens to sample
            model: the model to use
        """
        assert (
            is_openai_available
        ), "openai package is not installed or the API key is not set"
        super().__init__(temperature=temperature, max_tokens=max_tokens, model=model, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
    
    def __init_subclass__(cls, **kwargs):
        # check if the subclass has the required attributes
        for required in (
                "stateful",
                "type_name",
        ):
            if getattr(cls, required) is None:
                raise TypeError(
                    f"Can't instantiate abstract class {cls.__name__} without {required} attribute defined"
                )
        return super().__init_subclass__(**kwargs)
    
    # @retry(stop=stop_never, wait=wait_random_exponential(min=1, max=60))
    @retry(stop=stop_after_attempt(1), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages, azure_endpoint=None):
        if azure_endpoint == None:
            azure_endpoint = os.environ["END_POINT"]
        
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            max_retries=1,
            timeout=60
        )
        
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            # stop=STOP
        )
        response = completion.choices[0].message.content
        response = response.strip()
        return response
    
    def query(
            self,
            agent_name: str = None,
            role_desc: str = None,
            history_messages: List[Message] = None,
            global_prompt: str = None,
            request_msg: Message = None,
            azure_endpoint=None,
            *args,
            **kwargs,
    ) -> str:
        """
        Format the input and call the ChatGPT/GPT-4 API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request from the system to guide the agent's next response
        """
        # Merge the role description and the global prompt as the system prompt for the agent
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"{global_prompt.strip()}\n\n{role_desc}"
        else:
            system_prompt = role_desc
        
        if agent_name:
            all_messages = [(SYSTEM_NAME, system_prompt)]
            for msg in history_messages:
                if msg.agent_name == SYSTEM_NAME:
                    all_messages.append((SYSTEM_NAME, msg.content))
                else:  # non-system messages are suffixed with the end of message token
                    all_messages.append((msg.agent_name, f"{msg.content}"))
            
            if request_msg:
                all_messages.append((SYSTEM_NAME, request_msg.content + f" Now {agent_name} speaks."))
            else:  # The default request message that reminds the agent its role and instruct it to speak
                all_messages.append(
                    (SYSTEM_NAME, f"Now {agent_name} speaks.")
                )
            
            messages = []
            for i, msg in enumerate(all_messages):
                if i == 0:
                    assert (
                            msg[0] == SYSTEM_NAME
                    )  # The first message should be from the system
                    messages.append({"role": "system", "content": msg[1]})
                else:
                    if msg[0] == agent_name:
                        messages.append({"role": "assistant", "content": f"[{msg[0]}]: {msg[1]}"})
                    else:
                        if msg[0] == SYSTEM_NAME:
                            if messages[-1]["role"] == "user":
                                messages[-1]["content"] = f"{messages[-1]['content']}\n\n{msg[1]}"
                            else:
                                messages.append({"role": "user", "content": msg[1]})
                        elif messages[-1]["role"] == "user":  # last message is from user
                            messages[-1]["content"] = f"{messages[-1]['content']}\n[{msg[0]}]: {msg[1]}"
                        elif messages[-1]["role"] == "system" or messages[-1]["role"] == "assistant":
                            messages.append({"role": "user", "content": f"[{msg[0]}]: {msg[1]}"})
                        else:
                            raise ValueError(f"Invalid role: {messages[-1]['role']}")
        else:
            assert len(history_messages) % 2 == 1, 'The number of messages should be odd.'
            all_messages = [system_prompt] + [msg.content for msg in history_messages]
            if request_msg:
                all_messages[-1] += '\n' + request_msg.content
            
            messages = [{"role": "system", "content": all_messages[0]}]
            messages += [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, msg in
                         enumerate(all_messages[1:])]
        try:
            response = self._get_response(messages, azure_endpoint=azure_endpoint)
            # Remove the agent name if the response starts with it
            response = re.sub(rf"^\s*\[.*]:", "", response).strip()  # noqa: F541
            if agent_name:
                response = re.sub(
                    rf"^\s*{re.escape(agent_name)}\s*:", "", response, 1
                ).strip()  # noqa: F451
        except RetryError as e:
            err_msg = f"Agent {agent_name} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
            response = SIGNAL_END_OF_CONVERSATION + err_msg
        
        return response
    
    @abstractmethod
    async def async_query(
            self,
            agent_name: str,
            role_desc: str,
            history_messages: List[Message],
            global_prompt: str = None,
            request_msg: Message = None,
            *args,
            **kwargs,
    ) -> str:
        """Async querying."""
        raise NotImplementedError
    
    # reset the state of the backend
    def reset(self):
        if self.stateful:
            raise NotImplementedError
        else:
            pass
