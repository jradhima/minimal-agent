from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-lite-preview"
client = genai.Client()

# Initialize the conversation history
conversation = []

print("Chat with Gemini (type ctrl-c to quit)\n")

try:
    while True:
        # Get user input
        user_text = input("You: ")
        
        # Add user turn to conversation history
        conversation.append(types.Content(
            role="user", 
            parts=[types.Part.from_text(text=user_text)]
        ))

        # Send the entire conversation history to the model
        response = client.models.generate_content(
            model=MODEL,
            contents=conversation,
        )

        # Get the model's response
        model_text = response.text
        print("Agent:", model_text)
        print()

        # Add model turn to conversation history
        conversation.append(types.Content(
            role="model", 
            parts=[types.Part.from_text(text=model_text)]
        ))
except (KeyboardInterrupt, EOFError):
    print("\n\nAgent: Goodbye!")
