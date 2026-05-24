# Building a Minimal AI Agent with Google GenAI: An Educational Guide

## Introduction
* What is an AI Agent? (Brief, high-level definition: an AI that can *do* things, not just talk).
* The goal of this project: Building a "Minimal Viable Agent" from scratch.
* Tools needed: Python, `google-genai` SDK, and an API key.

## Part 1: The Simple Chat Interface
* Setting up the `genai` client.
* Creating a basic loop that sends user input to Gemini and prints the response.
* Why this is the "brain" of the agent before we give it "hands."

## Part 2: Introducing "Tools"
* The concept of Function Calling (giving the LLM a list of capabilities).
* Defining our first tools: `read_tool` and `list_tool`.
* How to register these tools so Gemini knows they exist.

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
