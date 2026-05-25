from google import genai

MODEL = "gemini-3.1-flash-lite-preview"
client = genai.Client()

# Initialize the chat session (automatically manages history)
chat = client.chats.create(model=MODEL)

print("Chat with Gemini (type ctrl-c to quit)\n")

try:
    while True:
        # Get user input
        user_text = input("You: ")

        # Send the message to the ongoing chat session
        response = chat.send_message(user_text)

        # Get and print the model's response
        model_text = response.text
        print("Agent:", model_text)
        print()

except (KeyboardInterrupt, EOFError):
    print("\n\nAgent: Goodbye!")
