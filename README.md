# Chatstack

## Minimalist Context Management for message-based GPTs

This Python code provides a chatbot implementation with context management using OpenAI's GPT-3.5-turbo or GPT-4 chat models. The chatbot maintains a conversation history and help manages the context state and size in tokens. 

### Dependencies

- loguru
- pydantic
- openai
- tiktoken

### OPEN_API_KEY

Chatstack finds your OpenAI API key via the OPENAI_API_KEY environment variable.

### Classes

- `ChatRoleMessage`: A base data class for messages with role, text, and tokens.
- `SystemMessage`: A data class for representing a message with the 'system' role.
- `ContextMessage`: A data class representing additional information context for the model.
- `AssistantMessage`: A data class for representing a message with the 'assistant' role .
- `UserMessage`: A data class for representing a message with the 'user' role.
- `ChatContext`: A class that manages the conversation context and generates responses using OpenAI message interface models.
- `ChatReponse`: A data class that contains the model response to a user message along with a record of the  input context sent to the model, and other significant details such as the model used, the number of tokens used, and the estimated cost of the request.

### Usage

1. Import the `ChatContext` class.
2. Create an instance of the `ChatContext` class with the desired configuration.
3. Call the `user_message` method with the user's message text to get a response from the chatbot.

Example:

```python
from chatstack import ChatContext

BASE_SYSTEM_PROMPT  = "You are a clever bot.  Do not apologize, or make excuses.  "
BASE_SYSTEM_PROMPT += "Do not mention that you are an AI language model since that is annoying to users."

def main():
    chat_context = ChatContext(base_system_msg_text=BASE_SYSTEM_PROMPT)

    print("Welcome to the Chatbot!")
    
    while True:
        user_input = input("You: ")      
        print("Chatbot:")
        response = chat_context.user_message(user_input, stream=True)
        

if __name__ == "__main__":
    main()
```


### Configuration

The `ChatContext` class accepts the following parameters:

- `min_response_tokens`: Minimum number of tokens to reserve for model completion response.
- `max_response_tokens`: Maximum number of tokens to allow for model completion response.
- `max_context_assistant_messages`: Number of recent assistant messages to keep in context.
- `max_context_user_messages`: Number of recent user messages to keep in context.
- `model`: The name of the GPT model to use (default: "gpt-3.5-turbo").
- `temperature`: The temperature for the model's response generation.
- `base_system_msg_text`: The base system message text to provide context for the model.

The primary method of the ChatContext is the user_message() which is used to assemble the input context to the model and generate a completion.

### `user_message(msg_text: str, stream: bool = False) -> ChatCompletionContext`

This method takes a user's message text as input and generates a response from the chatbot using the conversation context.

### `add_message(msg : ChatRoleMessage)`

Add a message to the context for presentation to the model in subsequent completion requests.

#### Parameters:

- `msg_text` (str): The text of the user's message.
- `stream` (bool, optional): If set to `True`, the response will be streamed to the console as it is generated. Default is `False`.

#### Returns:

- `ChatResponse`: An instance of the `ChatResponse` data class that includes the model response text, the actual input messages sent to the model, and other relevant details such as the token counts and estimated price of the completion.

