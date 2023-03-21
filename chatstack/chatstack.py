from loguru import logger
from pydantic import BaseModel, root_validator
from typing import Optional, List
from .retry import retry
import openai
import tiktoken
from time import time
import sys
from .pricing import price

encoder = tiktoken.get_encoding('cl100k_base')

class ChatRoleMessage(BaseModel):
    role: str
    text: str
    tokens: Optional[int]

    @root_validator
    def compute_tokens(cls, values) -> int:
        _text = f'{values["role"]}\n{values["text"]}' 
        values["tokens"] = len(encoder.encode(_text))
        values["tokens"] += 4    # per https://platform.openai.com/docs/guides/chat/managing-tokens
        return values


class SystemMessage(ChatRoleMessage):
    role = 'system'
    text: str

class ContextMessage(ChatRoleMessage):    
    """
    A message added to the model input context to provide context for the model.
    Generally contains source material from a search function
    Includes a prefix header to provide additional context to the model as to the nature of the content
    """
    role = 'system'
    prefix: str    
    text: str
    url: str

    def content(self) -> str:
        return f'{self.prefix}: {self.text}'
    
    @root_validator
    def compute_tokens(cls, values) -> int:
        _text = f'{values["role"]}\n{values["prefix"]}: {values["text"]}'
        values["tokens"] = len(encoder.encode(_text))
        values["tokens"] += 4    # per https://platform.openai.com/docs/guides/chat/managing-tokens
        return values    

    
class AssistantMessage(ChatRoleMessage):
    role = 'assistant'    
    text: str

class UserMessage(ChatRoleMessage):
    role = 'user'
    text: str


class ChatResponse(BaseModel):
    """
    returned in response to a User Message
    contains the complete state of the input context as sent to the model as well as the response from the model
    and other details such as the model used, the number of tokens used, and the estimated cost of the request
    """
    text:               str                             # the completion response from the model
    model:              str                             # the model used to generate the response
    temperature:        float                           # the temperature used to generate the response
    inputs:             List[ChatRoleMessage]           # the input context as sent to the model    
    input_tokens:       int                             # the number of tokens used in the input context
    response_tokens:    int                             # the number of tokens in the response text
    price:              float                           # the estimated price of the request in dollars


GPT3_MODELS = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301']
GPT4_MODELS = ['gpt-4', 'gpt-4-0314']

class ChatContext:

    def __init__(self,
                 base_system_msg_text : str = "Please respond to the user request.",  # base system message to use for context
                 min_response_tokens : int = 200,           # minimum number of tokens to reserve for model completion response;  
                 max_response_tokens : int = 400,           # maximum number of tokens to allow for model completion response; set to None to allow unlimited use of the model context
                 max_context_assistant_messages : int = 5,  # number of recent assistant messages to keep in context
                 max_context_user_messages : int = 50,      # number of recent user messages to keep in context
                 model : str = "gpt-3.5-turbo",             # model to use for completion
                 temperature : float = 0.5):                # temperature to use for model completion
                 
        self.model = model
        self.temperature = temperature
        self.base_system_msg = SystemMessage(text=base_system_msg_text)
        if model in GPT3_MODELS :
            self.max_model_context = 4096
        elif model in GPT4_MODELS:
            self.max_model_context = 8192
        else:
            raise ValueError(f"Model {model} not supported. Supported models are {GPT3_MODELS} and {GPT4_MODELS}")        
        self.min_response_tokens = min_response_tokens
        self.max_response_tokens = max_response_tokens
        self.max_context_assistant_messages = max_context_assistant_messages
        self.max_context_user_messages = max_context_user_messages
        self.messages = []    # reverse chronological order, newest messages first

    def _compose_completion_msg(self) -> List[ChatRoleMessage]:
        # assemble the input messages subject to the following constraints:
        # must leave room for min_response_tokens in the context
        # maximum of max_context_assistant_messages assistant messages
        # maximum of max_context_user_messages user messages
        max_input_context = self.max_model_context - self.min_response_tokens
        messages = []
        current_input_tokens = 2 + self.base_system_msg.tokens   # fixed overhead of 2 tokens per completion
        current_user_messages = 0
        current_assistant_messages = 0
        for msg in self.messages:
            tokens = msg.tokens
            if current_input_tokens + tokens > max_input_context:
                break
            if msg.role == 'assistant':
                if current_assistant_messages < self.max_context_assistant_messages:
                    messages.append(msg)
                    current_assistant_messages += 1
                    current_input_tokens += msg.tokens
            elif msg.role == 'user':
                if current_user_messages < self.max_context_user_messages:
                    messages.append(msg)
                    current_user_messages += 1
                    current_input_tokens += msg.tokens
        messages.append(self.base_system_msg)
        messages.reverse()
        return messages


    @retry(tries=10, delay=.05, ExceptionToRaise=openai.InvalidRequestError)
    def _completion(self, msgs :ChatRoleMessage, stream=False) -> str:

        messages = [{"role": msg.role, "content": msg.text} for msg in msgs]
        
        try:
            t0 = time()
            response =  openai.ChatCompletion.create(model = self.model,
                                                     messages = messages,
                                                     max_tokens = self.max_response_tokens,
                                                     temperature = self.temperature,
                                                     stream=stream)
        except openai.error.RateLimitError as e:
            logger.info(f'OpenAI RateLimitError, retrying...')
            raise
        except Exception as e:
            logger.warning(f'Exception during openai.ChatCompletion.create: {e}, retrying...')
            raise
        
        response_text = ""
        if stream:
            for chunk in response:
                output = chunk['choices'][0]['delta'].get('content', '')
                response_text += output
                sys.stdout.write(output)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            response_text = response['choices'][0]['message']['content']
        dt = time() - t0
        #logger.info(f'completion time: {dt:.3f} s')                                                    
        return response_text

    def add_message(self, msg : ChatRoleMessage):
        """
        Add a message to the context for presentation to the model.
        Does not result in a model completion.
        """
        self.messages.insert(0, msg)
        
    def user_message(self, msg_text : str, stream : bool =False) -> ChatResponse:
        msg = UserMessage(text=msg_text)
        # put message at beginning of list
        self.messages.insert(0, msg)
        completion_msgs = self._compose_completion_msg()
        response_text = self._completion(completion_msgs, stream=stream)
        response_msg = AssistantMessage(text=response_text)
        self.messages.insert(0, response_msg)
        
        input_tokens = sum([msg.tokens for msg in completion_msgs]) + 2            
        cr = ChatResponse(text = response_text,
                          model = self.model,
                          temperature=self.temperature,
                          inputs=completion_msgs,
                          input_tokens=input_tokens,
                          response_tokens=response_msg.tokens,
                          price=price(self.model, input_tokens, response_msg.tokens))
                                    
        return cr
    
        
