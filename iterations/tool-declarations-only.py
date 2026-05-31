from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-lite-preview"
client = genai.Client()


def list_tool_func(directory_path: str) -> str:
    """Lists all files and directories inside the specified path.

    Use this to see what files are available in a directory.
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

conversation = []

print("Chat with Gemini (type ctrl-c to quit)\n")

try:
    while True:
        user_text = input("You: ")

        conversation.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=conversation,
            config=types.GenerateContentConfig(tools=[agent_tools]),
        )

        if response.text:
            print("Agent:", response.text)
        else:
            print("Agent: (Thinking... or tried to call a tool that wasn't executed)")
            print("Raw response:", response.candidates[0].content.parts)
        print()

        conversation.append(response.candidates[0].content)
except (KeyboardInterrupt, EOFError):
    print("\n\nAgent: Goodbye!")
