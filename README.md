# Email Assistant

A command-line email assistant built with LangChain's `create_agent`. It shows three agent patterns working together: gating tools behind a verification step, pausing for human approval before a risky action, and tracing every run with LangSmith.

The inbox and the email sending are fake (hardcoded) so the focus stays on the agent logic.

## Features

- **ID verification:** The assistant asks for a user ID first. A custom middleware (`authenticate_user`) only exposes the email tools after the saved ID matches. Before that, the model can only call `save_user_id`.
- **Human-in-the-loop approval:** `send_email` pauses the agent with `HumanInTheLoopMiddleware`. The script shows the recipient, subject and body, and you type `y` or `n` before anything is sent. A rejection can include a reason that goes back to the model.
- **Custom state:** The agent keeps `user_id` in its own state schema, and `InMemorySaver` keeps the conversation going across turns on one thread.
- **Tracing:** With LangSmith enabled, every model call, tool call and approval pause shows up as a trace.

## Setup

You need Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/patrkdev/email-assistant.git
cd email-assistant
uv sync
```

Create a `.env` file next to `email_assistant.py` (copy `.env.example`):

```
OPENAI_API_KEY=your-openai-key

# optional, for tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=email-assistant
```

## Run

```bash
uv run email_assistant.py
```

Type `quit` to exit.

## Example session

```
Assistant: Hi there! Welcome. Could you share your user ID?
You: my id is 456
Assistant: That ID wasn't accepted. Please try again.
You: it might be <valid id>
Assistant: Welcome back! Your ID is verified. What would you like to do?
You: what does my email say?
You: send a reply confirming I'll be there

--- Approval needed ---
To:      john@example.com
Subject: Re: 9am-5pm tomorrow
Approve this? (y/n):
```

The valid ID is the `VALID_ID` constant at the top of `email_assistant.py`.

## How it works

1. `authenticate_user` runs before every model call and swaps the tool list based on `user_id` in state. The access check lives in code, not in the prompt.
2. `save_user_id` checks the ID and returns "accepted" or "rejected", so the model repeats a result instead of guessing.
3. `HumanInTheLoopMiddleware` interrupts on `send_email`. The script resumes the same thread with `Command(resume={"decisions": [...]})`.

## Project files

| File | Purpose |
| --- | --- |
| `email_assistant.py` | The agent and the chat loop |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for the environment variables |

## Built with

LangChain, LangGraph, OpenAI (`gpt-5-nano`), LangSmith.