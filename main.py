from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-lite-preview"


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


def write_tool_func(filepath: str, content: str) -> str:
    """Creates and writes the contents of a file.

    Use this to create a file. It will create the directories needed.
    Returns the relative filepath or a specific error message if it fails.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(filepath)

    except IsADirectoryError:
        return f"Error: '{filepath}' is a directory path, not a file path."
    except PermissionError:
        return f"Error: Permission denied. You do not have access to write to '{filepath}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"


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

write_tool_decl = types.FunctionDeclaration(
    name="write_tool_func",
    description="Creates or overwrites a file with the specified content. Automatically creates parent folders if they are missing.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "The system path where the file should be created or updated.",
            },
            "content": {
                "type": "string",
                "description": "The exact string content to write into the file.",
            },
        },
        "required": ["filepath", "content"],
    },
)

agent_tools = types.Tool(
    function_declarations=[read_tool_decl, list_tool_decl, write_tool_decl]
)
tool_map = {
    "read_tool_func": read_tool_func,
    "list_tool_func": list_tool_func,
    "write_tool_func": write_tool_func,
}


class Agent:
    def __init__(self, client: genai.Client):
        self.client = client
        self.conversation = list()

    def get_user_response(self) -> types.UserContent:
        msg = input("You: ")
        print()
        return types.UserContent(parts=types.Part.from_text(text=msg.strip()))

    def get_model_response(self) -> types.GenerateContentResponse:
        return self.client.models.generate_content(
            model=MODEL,
            contents=self.conversation,
            config=types.GenerateContentConfig(
                tools=[agent_tools],
                system_instruction=(
                    "You're a helpful programming agent. ",
                    "CRITICAL EFFICIENCY RULE: Whenever you discover multiple files that need to be read or inspected or written, "
                    "you MUST execute all function calls in PARALLEL within a single turn. DO NOT call them one-by-one.",
                ),
            ),
        )

    def resolve_tool_call(self, tool_call: types.FunctionCall) -> types.Part:
        print(f" -> Executing: {tool_call.name} with args {tool_call.args}...")
        try:
            if func := tool_map.get(tool_call.name, None):
                function_result = func(**tool_call.args)
            else:
                function_result = f"Error: Tool '{tool_call.name}' is not recognized."
            function_response = {"result": function_result}
        except Exception as e:
            function_response = {"error": str(e)}

        return types.Part.from_function_response(
            name=tool_call.name,
            response=function_response,
        )

    def resolve_tool_calls(
        self, response: types.GenerateContentResponse
    ) -> types.Content:
        print(f"\n[Agent requested {len(response.function_calls)} parallel tool calls]")
        function_response_parts = []
        for call_part in response.function_calls:
            function_response_parts.append(self.resolve_tool_call(call_part))

        function_response_content = types.Content(
            role="tool", parts=function_response_parts
        )
        return function_response_content

    def run(self):
        print("Chat with Gemini (type ctrl-c to quit)\n")

        try:
            while True:
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
                else:
                    print("Agent: Bye!")
                    break
                print()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAgent: Goodbye!")


def main():
    client = genai.Client()
    agent = Agent(client=client)

    agent.run()


if __name__ == "__main__":
    main()
