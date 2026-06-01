# Building a Minimal AI Agent

> The code in this article can be found in [this repo](https://github.com/jradhima/minimal-agent)

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
* Attatched info: Available tools: [`list_tool_func`, `read_tool_func`, ...], you can call them this way, ...

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
```
>> uv run iterations/tool-declarations-only.py
Chat with Gemini (type ctrl-c to quit)

You: Can you tell me how many files are in this repo?
Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
Agent: (Thinking... or tried to call a tool that wasn't executed)
Raw response: [Part(
  function_call=FunctionCall(
    args={
      'directory_path': '.'
    },
    id='aWXnr3fm',
    name='list_tool_func'
  ),
  thought_signature=b'\x124\n2\x01\x0c9\xd6\xc7\xee\xd4\xb2\xefC\xea(\xa8T=T\xd7AsP7gXj\n\xd7E\xf4\x83\xdc\xa7\xfe\xd6\xd5|\x8ckw\x84\xe0\x8f\xde<\xe5g\xd3\x16\xee\xbcj'
)]

You:
```
The model tried to do something! We asked it to count files and it (correctly) started by trying to see how many there are. LLMs are nowdays post trained using reinforcement learning on these types of conversations/patterns A LOT, so they naturally know what tools are and how to call them. You don't need to explain much or write a prompt explaining how to, they just know.

We are now in a good spot, what's left is helping the LLM actually use the tools. An LLM can only tell us what it wants to do, it's up to us to give it "hands" with which to use the tools. The LLM "throws" and we must implement the "caching" part.

## Part 3: Giving the Agent "Hands" (Tool Execution)

### Function calls
In the logs shown above, we can see a `FunctionCall` object. This is part of the response the model returns. When we declare tools in our request, the LLM may respond with a normal response or it may respond with a response *plus* function calls that it wants us to resolve. Basically, the LLM might say something like:

>In order to tell you how many files exist in the repo, i first need to look there myself. I see that there is a `list_tool` available and it returns a list of filepaths which is exactly what I need. The user asked for the current repo which means i should ask for path `.`

It will then format its response in such a way that the library we use can intercept the response and serialize it in a structured way. Part of this will be the model response while part of it will be the requested function calls. To fetch these we can just reach for them:
```python
response = client.models.generate_content(
    model=MODEL,
    contents=conversation,
    config=types.GenerateContentConfig(tools=[agent_tools]),
)

print(response.function_calls)
```
This data structure contains a list of everything we need. Function names and arguments. There is 1 minor problem, we need to write down a way to go from strings to function calls. The simplest way to do this would be something like:
```python
for call in response.function_calls:
    if call.name == "list_tool_func":
        result = list_tool_func(**call.args)
    elif call.name == "read_tool_func":
        result = read_tool_func(**call.args)
    else:
        result = f"Error: Tool '{call.name}' is not recognized."
```
In a more modern take, it could look like this but it's still not super nice:
```python
for call in response.function_calls:
    match call.name:
        case "list_tool_func":
            result = list_tool_func(**call.args)
        case "read_tool_func":
            result = read_tool_func(**call.args)
        case _:
            result = f"Error: Tool '{call.name}' is not recognized."
```
A better pattern is something like this:
```python
tool_map = {
    "read_tool_func": read_tool_func,
    "list_tool_func": list_tool_func,
}

for call in response.function_calls:
    if func := tool_map.get(call.name, None):
        result = func(**call.args)
    else:
        result = f"Error: Tool '{call.name}' is not recognized."
```
So now we get a better idea of the process. We must have predefined the *plumbing* to get text wired to the correct functions to execute. Without this trick, the LLM responds with text but this leaves us with an ugly looking chat or, at best, with the agent guiding us how to be agents ourselves (if we assume we follow its instructions to achieve our goal)!

### Giving control to the agent
One important thing to note is that a tools flow is different to a chat flow in an important way; when responding in a chat, the LLM will give control back to us because it has been trained to generate tokens until it reaches the end of what it has to say (actually, until the token for "terminate discussion" is the best candidate for next token to use). So a conversation is incremented by 1 party at a time, once us, then the model, then us, then the model...

However with tools, the proper flow is different; the model gets to choose when it gives control back to us. We ask something, the model "thinks" and at face value, the process is the same. It terminates with the agent sending its response. However this response assumes that we will run the function calls and send back the results so the LLM can resume working. We could choose to not do this. We could route the function call results to us, write our response, then send a new request. This is suboptimal. The agent should be able to run independently until it thinks it has a response. This means that we trust it to enter a tool calling loop until it's ready. This looks something like this:
```python
self.conversation.append(self.get_user_response())
response = self.get_model_response()

while True:
    if response.candidates:
        self.conversation.append(response.candidates[0].content)

    if not response.function_calls:
        break

    tool_content = self.resolve_tool_calls(response)
    self.conversation.append(tool_content)
    response = self.get_model_response()

if response.candidates and response.text:
    print("Agent: ", response.text)
```
We can add guardrails here, like a maximum number of tool calls. It's up to us, however for the agent to be most effective it must be able to work independently.

## Part 4: Putting It All Together
We've covered the theory, the tools, and the plumbing. Now it's time to assemble everything into a working agent. If you look at `main.py` in the linked repo, there is an `Agent` class that handles the lifecycle of a request, from user input to the final response. There are also 2 more tools, `edit_tool` and `write_tool`, just to make things more interesting.

### The Agent Class
The `Agent` class is the orchestrator. Its main job is to maintain the conversation state and handle the "handshaking" with the Gemini model.

```python
class Agent:
    def __init__(self, client: genai.Client):
        self.client = client
        self.conversation = list()
```

By keeping `self.conversation` as a list, we maintain the full history of the interaction—including the back-and-forth between the user, the model, and the tool results.

### The Coordination Loop
The heart of the agent is the `run()` method, which handles the flow when an agent decides to use tools.

```python
def run(self):
    # ...
    while True:
        self.conversation.append(self.get_user_response())
        response = self.get_model_response()

        # The inner loop: tool calling
        while True:
            if response.candidates:
                self.conversation.append(response.candidates[0].content)

            if not response.function_calls:
                break

            tool_content = self.resolve_tool_calls(response)
            self.conversation.append(tool_content)
            response = self.get_model_response()
```

The nested `while True` loop is exactly what we were talking about in the previous section: the agent might need to call a tool, wait for the result, think, and call *another* tool. It might repeat this cycle several times before it finally has enough information to give its final answer.

### Parallel Execution
One of the most important pieces for the understanding of how agents work is the instruction we give the model in the `system_instruction`:

> "CRITICAL EFFICIENCY RULE: Whenever you discover multiple files that need to be read or inspected or written, you MUST execute all function calls in PARALLEL within a single turn."

Because we structured our `Agent` to handle multiple function calls in `resolve_tool_calls`, the agent can indeed run operations in parallel. This is a massive performance boost when scanning an entire directory or reading multiple related files.

I added this specific efficiency rule after consulting with Gemini about why my agent would not perform parallel calls! I would ask it to tell me about the files in the repo, then the agent would list read them 1 at a time even though it knows that it can send multiple function calls in 1 response! Sometimes you have to resort to tricks like these to make agents listen and follow instructions. A lot of this has to do with how providers perform post-training but not all models are equally good at following instructions or using tools. General LLM "intelligence" is one thing, agentinc workflow performance is a different topic.

### Running the Agent
Now that the tools (`list`, `read`, `write`, `edit`) are fully registered and the `Agent` is orchestrating the flow, go ahead and run it:

```bash
uv run main.py
```

You can now ask the agent questions like: *"List the files in this directory"* or *"Read main.py and tell me how the Agent class works"*. You will see the agent "thinking," identifying the correct tool, executing it, and integrating the results into its reasoning—all without you lifting a finger.

## Part 5: Using the SDK fully
We've done everything but the code seems a bit lengthy, filled with boilerplate and lines of code that are really not interesting at all! Thankfully all of this was intentional to make the agent as explicit as possible.

In the linked repo, `iterations/05-final-agent.py` demonstrates how to achieve the same functionality in under 150 lines. 

### Automatic Tool Registration
Instead of manually defining `types.FunctionDeclaration` for every tool, the SDK can reflect on Python functions directly. If you provide a function with a well-formatted docstring, the SDK handles the conversion for you:

```python
# The SDK automatically extracts the schema from the function and docstring
chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(
        tools=[list_tool_func, read_tool_func, write_tool_func, edit_tool_func]
    )
)
```

### Automatic Tool Resolution
The `chat` object is actually "tool-aware." When you call `chat.send_message()`, it can automatically execute the functions requested by the model and return the final text result, effectively collapsing your entire `while` loop into a single function call.

Actually, the tool-awareness is part of the SDK and is enabled by default if we pass the functions to the request directly. This part comes straight from the docs:
```python
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return 'sunny'


response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
    ),
)

print(response.text)
```
We had to implement it because we were doing things "the hard way" (nothing hard about it, just boring!) but generally these things are taken care off. That said, using the manual way allowed us to learn more, to debug specific agent behaviors and to implement non-standard control (like injecting specialized logging). There may be ways to do these things using the SDK as well but in general, this is the standard compromise when using 3rd party code and higher level abstractions.


## Conclusion
Next steps? Actual coding agents implement memory to handle conversation serialization so the agent doesnt't "forget" (could be simply a directory with JSON files, Opencode used this for a *long* time until they switched to sqlite), web search, a bash tool, subagents. What about security? This agent could overwrite all our files and if it had a bash tool could just `rm -rf /` our computer to oblibion! A hot topic is sandboxing and how to make sure the agent can work freely without the user having to babysit every tool call (coding agents can ask for user permission on tool calls).

The choices are limitless but in many ways they are not really! We started with a basic API request and finished with a functional agent capable of basic interactions with the file system. An agent that doesn't just talk, but acts. This project was meant to show that building an "AI Agent" isn't magic at all. In the end, it's just plumbing, loops and a good LLM.
