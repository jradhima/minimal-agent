# Minimal Agent

This repository contains a simple, interactive AI agent built with Python and the Google GenAI SDK. 

The agent is designed to assist with file manipulation tasks within the local file system. It features built-in tools for listing directories, reading files, creating files, and editing specific blocks of text within files.

## Features

- **Interactive Shell**: Communicate with Gemini directly from your terminal.
- **File System Tools**:
    - `list_tool_func`: List files and folders in a directory.
    - `read_tool_func`: Read the contents of a file.
    - `write_tool_func`: Create or overwrite files.
    - `edit_tool_func`: Modify specific text blocks within existing files.
- **Parallel Tool Execution**: The agent is configured to handle multiple tool requests in parallel, improving efficiency.

## Getting Started

1.  **Prerequisites**: Ensure you have Python 3.13+ and [uv](https://github.com/astral-sh/uv) installed.
2.  **Installation**:
    ```bash
    uv sync
    ```
3.  **Run the Agent**:
    ```bash
    uv run main.py
    ```

*Note: Ensure your `GOOGLE_API_KEY` environment variable is set before running the agent.*
