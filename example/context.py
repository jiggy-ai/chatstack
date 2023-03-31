from chatstack import ChatContext, ContextMessage, AssistantMessage
import sys
from random import randint

BASE_SYSTEM_PROMPT  = "You are a clever bot.  Do not apologize, or make excuses.  "
BASE_SYSTEM_PROMPT += "Help the user count the context messages. The number of context messages changes every time."

def main():
    chat_context = ChatContext(base_system_msg_text=BASE_SYSTEM_PROMPT)

    print("Welcome to the Chatbot! ")

    # add a context message
    
    
    while True:
        dc = [ContextMessage(text=f"context message {i}", prefix="context message") for i in range(randint(200, 400))]
        print(f"generated {len(dc)} context messages and added them to the context.  Ask the bot about them.")
        user_input = input("You: ")      
        print("Chatbot:")
        chat_rsp = chat_context.user_message(user_input, dynamic_context=dc)
        print("Chatbot: ", chat_rsp.text)
        user_messages = len([m for m in chat_rsp.inputs if m.role == 'user'])
        context_messages = len([m for m in chat_rsp.inputs if isinstance(m, ContextMessage)])
        assistant_messages = len([m for m in chat_rsp.inputs if isinstance(m, AssistantMessage)])
        print(f"({user_messages} user messages, {context_messages} context messages, {assistant_messages} assistant messages")
        # print token usage and price
        print(f"({chat_rsp.input_tokens + chat_rsp.response_tokens} tokens (${chat_rsp.price:.4f}))")
        
if __name__ == "__main__":
    main()
