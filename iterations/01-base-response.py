from google import genai

# Initialize the client (automatically uses GEMINI_API_KEY from environment)
client = genai.Client()

# Send a simple request
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents="Hello, give me a very short introduction about yourself.",
)

print(response.text)
