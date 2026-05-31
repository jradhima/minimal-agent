# Building a Minimal AI Agent

## Introduction
If you're a dev of any type and you're not living under a rock in 2026, you've surely used or at least heard about AI agents. I will only mention my dislike for the use of "AI" because I don't believe the "I" part but let's continue.

What is an AI Agent? Put briefly, it's an AI system that can *do* things, not just talk. You go to chatgpt.com and chat with Mr.GPT himself but he is not at his most powerful. In fact, he used to be much weaker. Now he can browse the web and even run some calculations, in the past he could only talk back.

Turns out that the difference between an AI chat and an agent is very minimal. You only need 1 idea to make it work and the implementation is very simple. The goal of this project is to build a "Minimal Viable Agent" from scratch. We will do it step by step, from the most basic iteration to a more complete one.

The inspiration for this article or potentialy series is one of my agents of choice, [pi](https://pi.dev). To follow the article, the tools needed are: Python and an API key. I will start by using the Google GenAI SDK and a Gemini API key but the choice shouldn't really matter, it's pretty much the same across all providers. To get a Gemini API key, head to [Google AI studio](https://aistudio.google.com/) and create one for free.

## Part 1: The Simple Chat Interface

### Connect to the API
First things first, we need to make sure that we can connect to the LLM. An agent is basically a program that realizes a continuous, back-and-forth flow between the user and the LLM. To make it work, we will need continuous on-demand access to the LLM of choice.

Run the following commands to set up your environment:

```bash
# Initialize the project and add dependencies
uv init
uv add google-genai

# Export your API key (replace with your actual key)
export GEMINI_API_KEY="your_actual_api_key_here"
```

Create a `main.py` file to verify the connection:

```python
from google import genai
import os

# Initialize the client (automatically uses GEMINI_API_KEY from environment)
client = genai.Client()

# Send a simple request
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents="Hello, give me a very short introduction about yourself.",
)

print(response.text)
```

Now, run it with `uv run main.py`. You should see a response from the model. If not, something is wrong with the API key, the model selected or with the virtual environment.

```
>> uv run base-request-response.py
Hello! I am an AI assistant trained by Google. I’m here to help you answer questions, draft content, solve problems, and explore ideas quickly and efficiently. How can I help you today?
```

### Setting up the chat the "hard" way
Most LLM APIs are stateless, meaning they treat each request independently. This goes against the idea of a conversation, which is a series of responses in chronological order, so what gives? Well, there is a very simple, brute-force solution: each turn we send the entire conversation to the LLM.

I could say "each turn we have to remind the LLM of the entire conversation" but *remind* is really the wrong word; each turn we have to *stage* the conversation for the LLM to respond. What happens whenever you converse with an LLM is that on each turn, the LLM sees something that looks like a language fill-the-gap exercise:
- user message 1
- response 1
- user message 2
- (LLM fills in the appropriate response)

Given this, the clear next step for our minimal agent is to reach the status of *chatbot*. We achieve this by putting the request-response in a while loop:

```python
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
```

This a basic loop that sends user input to Gemini and prints the response is the backbone of all agents, including CC or Codex. They all work in this way, building a conversation and sending it back and forth. Recently, OpenAI introduced a websocket mode to keep persistent connections which allows sending only the latest part of a conversation but the large majority of LLM interaction still happens with the stateless mode.

Testing the new code and it seems to be working!
```
>> uv run iterations/chat-loop.py
Chat with Gemini (type ctrl-c to quit)

You: hi
Agent: Hello! How can I help you today?

You: is a python dict a hash map? respond in 10 words or less.
Agent: Yes, a Python dictionary is implemented as a hash map.

You:
```

### Setting up the chat the "easy" way
As mentioned, the work we did previously was setting up things the "hard" way. Nothing hard about it but in every turn we are doing the same thing; create the user part, add it to the conversation, send the request, fetch the response, add it to the conversation, etc... There is an easy abstraction that can simplify the process, in the form of a "conversation" class. Because going back-and-forth with the agent is so common, I will guess that all providers have added something similar to their SDKs. Google definitely has and it makes our code much nicer to look at.

```python
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
```

The boilerplate of creating parts and assigning a user or model role, plus the manual management of the conversation is gone. This is how you would create an actual agent but for reasons of clarity, I will continue doing things the *manual* way in order to be as explicit as possible. It will be clear why in the next section, where we introduce *tools*.

## Part 2: Introducing "Tools"

### Tools
So far our agent is very cool but it's not really an agent. It doesn't have agency, meaning it cannot interact with the environment around it. It can only talk and hope we copy it's response and use it. The next step is to give it tools that will allow it to *do* things.

Obviously when writing code, you cannot really give the agent a wrench or a screwdriver. Our tools will be functions, normal everyday functions that the agent will be able to call. For example, what might an agent want to do? He probably wants a way to see. Seeing in this case means listing and reading files. Let's define our first tools: `list_tool` and `read_tool`.

### The functions
Let's define a minimal function that will allow our agent to list files. It probably looks like this:
```python
def list_tool_func(directory_path: str) -> str:
    """Lists all files and directories inside the specified path.

    Do not use this with files, only with directories.
    Returns a newline-separated string of items, or an error message if it fails.
    """
    try:
        path = Path(directory_path)

        if not path.exists():
            return f"Error: The directory '{directory_path}' does not exist."
        if not path.is_dir():
            return f"Error: '{directory_path}' is a file, not a directory. Use read_tool instead."

        items = [item.name for item in path.iterdir()]

        if not items:
            return "Directory is empty."
        return "\n".join(items)

    except PermissionError:
        return f"Error: Permission denied. You do not have access to view '{directory_path}'."
    except Exception as e:
        return f"Error listing directory: {str(e)}"
```

Now that we have the code to list files, let's also create code that can read files:
```python
def read_tool_func(filepath: str) -> str:
    """Reads and returns the contents of a file.

    Use this to read the contents of a file. Do not attempt to use this with directories.
    Returns the file content string, or a specific error message if it fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return "Error: File is empty."
        return content

    except FileNotFoundError:
        return f"Error: The file '{filepath}' does not exist. Check the path and try again."
    except IsADirectoryError:
        return f"Error: '{filepath}' is a directory, not a file. Use list_tool instead."
    except PermissionError:
        return f"Error: Permission denied. You do not have access to read '{filepath}'."
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

These are some very crude implementations, especially the read tool. It returns the full content of a file which is dangerous for 2 main reasons:
* A very large file can cause memory issues
* More importantly, loading the full file will add all of it to the context of the conversasion, polluting the context window and reducing the efficiency of the agent by a lot

For these and lots of other reasons, an actual read tool implementation would have offsets and truncation plus a lot of other features to make it more "industrialized" but that level of detail is beside the point of this post.

So now that we have the tools, what do we see? They are just functions like any other function! Now our agent should be ready to list and read files, no?

### Eeeeh.. nothing happened..?
Well, not really. Just running the code again will not do anything. There is nothing actually linking the tools with the agent, they are just standing there, dead code. Your linter is probably complaining that they're not used and can be removed!

The clear solution is to declare these tools to the agent, let him know of their existence. It may sound crazy but it's true,  this is how tool calling actually works! You just tell the agent about the possible tools the LLM can use. You have to be very specific about them but the good thing is that LLMs are very good at reading and "understanding" code. So basically, the start of the conversation will look *something* like this:
* You: Hey, can you tell me what this repo does?
* Attatched info: Available tools: [`list_tool`, `read_tool`, ...], you can call them this way, ...

### Registering the tools
Let's see how you register the tools. We will first do it the *manual* way. We must write some function declarations which are basically the user manual of the tools in a JSON format.
```python
list_tool_decl = types.FunctionDeclaration(
    name="list_tool_func",
    description="Lists the names of all files and folders inside a specified directory. Returns an explicit error message if the path doesn't exist or is a file.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "The relative or absolute system path of the directory/folder you want to inspect.",
            }
        },
        "required": ["directory_path"],
    },
)

read_tool_decl = types.FunctionDeclaration(
    name="read_tool_func",
    description="Reads and returns the complete text contents of a specific file. Returns an explicit error message if the file doesn't exist, is empty, or is a directory.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "The relative or absolute system path of the file to read. Do not use this tool on directories.",
            }
        },
        "required": ["filepath"],
    },
)

agent_tools = types.Tool(function_declarations=[read_tool_decl, list_tool_decl])
```
Having defined the declarations, we must pass them to the agent with the request. We do this by modifying the request code as such:
```python
response = client.models.generate_content(
            model=MODEL,
            contents=conversation,
            config=types.GenerateContentConfig(tools=[agent_tools])
)
```
Let's try and see if it's going to work!

## Part 3: Giving the Agent "Hands" (Tool Execution)
* The logic behind `resolve_tool_calls`.
* Mapping JSON function calls from the model back to Python functions.
* The "Parallel Execution" rule: Handling multiple tool requests efficiently.

## Part 4: Putting It All Together
* Walking through the full `main.py` flow.
* How the `Agent` class coordinates the chat loop and the tool loop.
* Running the agent for the first time.

## Conclusion
* Recap of what was built.
* Where to go from here: Adding memory, search capabilities, or web browsing.
