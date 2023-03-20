# Chatstack

## Minimalist Context Management for message-based GPTs

This Python code provides a chatbot implementation with context management using OpenAI's GPT-3.5-turbo or GPT-4 chat models. The chatbot maintains a conversation history and manages the context to ensure meaningful responses.

### Dependencies

- loguru
- pydantic
- openai
- tiktoken

### OPEN_API_KEY

Chatstack finds your OpenAI API key via the OPENAI_API_KEY environment variable.

### Classes

- `ChatRoleMessage`: A base class for messages with role, text, and tokens.
- `SystemMessage`: A message with the role 'system'.
- `ContextMessage`: A message added to the model input context to provide context for the model.
- `AssistantMessage`: A message with the role 'assistant'.
- `UserMessage`: A message with the role 'user'.
- `ChatContext`: A class that manages the conversation context and generates responses using OpenAI message interface models.

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
        response = chat_context.user_message(user_input, stream=True)
        print("Chatbot:", response)

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
