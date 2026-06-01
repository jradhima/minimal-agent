from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-lite-preview"

# --- Tools Definition ---


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
        return "\n".join(items) if items else "Directory is empty."
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
        return content if content.strip() else "Error: File is empty."
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


def edit_tool_func(filepath: str, old_text: str, new_text: str) -> str:
    """Edits an existing file by replacing a specific block of text.

    Use this to modify specific sections of a file without rewriting the whole thing.
    Returns a success message with the filepath, or an error message if it fails.
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return f"Error: The file '{filepath}' does not exist. Use write_tool to create it."
        if path.is_dir():
            return f"Error: '{filepath}' is a directory, not a file."

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if old_text not in content:
            return (
                f"Error: Could not find the exact text block to replace in '{filepath}'. "
                "Ensure your 'old_text' matches the file content exactly, including spacing and indentation."
            )

        with open(path, "w", encoding="utf-8") as f:
            f.write(content.replace(old_text, new_text))

        return f"Successfully updated '{filepath}'."
    except PermissionError:
        return f"Error: Permission denied. Cannot modify '{filepath}'."
    except Exception as e:
        return f"Error editing file: {str(e)}"


# Clean array of python functions
tools_list = [read_tool_func, list_tool_func, write_tool_func, edit_tool_func]


# --- Main Execution ---


def main():
    client = genai.Client()

    # Create the chat session with automatic tools and system instruction attached
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            tools=tools_list,  # Automatically creates schemas & binds function local execution!
            system_instruction=(
                "You're a helpful programming agent. "
                "CRITICAL EFFICIENCY RULE: Whenever you discover multiple files that need to be read or inspected or written, "
                "you MUST execute all function calls in PARALLEL within a single turn. DO NOT call them one-by-one."
            ),
        ),
    )

    print("Chat with Gemini (type ctrl-c to quit)\n")

    try:
        while True:
            user_text = input("You: ")
            if not user_text.strip():
                continue

            # chat.send_message automatically checks if Gemini wants to call a tool,
            # executes the local Python code, handles the response loop, and updates history.
            response = chat.send_message(user_text)

            print("Agent:", response.text)
            print()

    except (KeyboardInterrupt, EOFError):
        print("\n\nAgent: Goodbye!")


if __name__ == "__main__":
    main()
