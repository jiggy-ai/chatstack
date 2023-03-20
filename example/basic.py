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
        # print token usage and price
        print(f"({response.input_tokens + response.response_tokens} tokens (${response.price:.4f}))")
if __name__ == "__main__":
    main()
