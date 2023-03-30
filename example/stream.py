from chatstack import ChatContext
import sys

BASE_SYSTEM_PROMPT  = "You are a clever bot.  Do not apologize, or make excuses.  "
BASE_SYSTEM_PROMPT += "Do not mention that you are an AI language model since that is annoying to users."

def main():
    chat_context = ChatContext(base_system_msg_text=BASE_SYSTEM_PROMPT)

    print("Welcome to the Chatbot!")
    
    while True:
        user_input = input("You: ")      
        print("Chatbot:")
        for chat_rsp in chat_context.user_message_stream(user_input):
            sys.stdout.write(chat_rsp.delta)
            sys.stdout.flush()
        # print token usage and price
        print(f"({chat_rsp.input_tokens + chat_rsp.response_tokens} tokens (${chat_rsp.price:.4f}))")
        
if __name__ == "__main__":
    main()
