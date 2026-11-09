from typing import Callable
from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

VALID_ID = "007"


# custom state
class EmailState(AgentState):
    user_id: str


# tools
@tool
def save_user_id(user_id: str, runtime: ToolRuntime) -> Command:
    """Save the user ID that the user types in the chat. Pass only the ID value itself,
    not the rest of their sentence."""
    user_id = user_id.strip()
    if user_id == VALID_ID:
        result = "ID accepted. The user is verified."
    else:
        result = "ID rejected. It is not correct. Ask the user to try again."

    return Command(update={
        "user_id": user_id,
        "messages": [ToolMessage(result, tool_call_id=runtime.tool_call_id)],
    })


@tool
def read_email() -> str:
    """Read the users email"""
    return """
    Hi Patrick,

    Can you come in for a 9am to 5pm tomorrow?

    Best,
    John.
    (john@example.com)
    """


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email with the given subject body."""
    # fake email sending
    return f"Email sent to {to} with subject {subject} and body {body}"


# middleware: only expose email tools once the ID is valid
@wrap_model_call
def authenticate_user(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Dynamically pick tools based on the user's id in the state"""
    user_id = request.state.get("user_id")

    if user_id == VALID_ID:
        tools = [read_email, send_email]  # verified
    else:
        tools = [save_user_id]  # no or wrong ID

    return handler(request.override(tools=tools))


system_prompt = """
You are the user's friendly email assistant. You're warm and easygoing, but you keep things quick and don't waste the user's time.

Verification:
- At the start of the conversation, greet the user warmly and ask for their user ID. Do nothing else until they give it.
- When the user gives their ID, call save_user_id with only the ID itself, taken from their message. Drop any surrounding words like "my id is" or "my bad". Never change or guess the ID.
- After save_user_id runs, read its result.
- If it says the ID was accepted, welcome the user and ask what they'd like to do. You can only read their email and send emails. Never offer or imply any other capability.
- If it says the ID was rejected, tell the user in a friendly way that the ID they entered was not correct, then ask them to try again. Never hint at the correct ID.

Reading email:
- Call read_email. If it's short, show it as is. If it's long, summarize the key points instead of showing the whole thing.

Sending email:
- Write a complete draft (to, subject, body) and show it to the user so they can see exactly what will go out.
- Call send_email with that draft right away. Do not ask "should I send this?" or wait for a yes first. The system has its own separate approval step before anything actually sends, so asking again is redundant.
- Only tell the user the email was sent if send_email actually returned success.

General:
- Keep replies short, direct, and friendly. No filler.
- Never read or send email before the ID is accepted.
"""

agent = create_agent(
    model="gpt-5-nano",
    tools=[save_user_id, read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
    system_prompt=system_prompt,
    middleware=[
        authenticate_user,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "save_user_id": False,
                "read_email": False,
                "send_email": True,
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
)

config = {"configurable": {"thread_id": "1"}}


def ask_for_decisions(interrupt) -> list[dict]:
    """Show each pending tool call and collect an approve/reject decision."""
    decisions = []
    for action in interrupt.value["action_requests"]:
        args = action["args"]
        print("\n--- Approval needed ---")
        print(f"Tool:    {action['name']}")
        print(f"To:      {args.get('to')}")
        print(f"Subject: {args.get('subject')}")
        print(f"Body:\n{args.get('body')}\n")

        while True:
            answer = input("Approve this? (y/n): ").strip().lower()
            if answer in ("y", "yes"):
                decisions.append({"type": "approve"})
                break
            if answer in ("n", "no"):
                reason = input("Why reject? (optional): ").strip()
                decisions.append({
                    "type": "reject",
                    "message": reason or "The user rejected this action.",
                })
                break
            print("Please type y or n.")
    return decisions


def run_turn(user_text: str) -> None:
    """Send one user message, handle any approval pauses, print the reply."""
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
    )

    # While the agent is paused waiting for approval, resume it with a decision
    while "__interrupt__" in response:
        decisions = ask_for_decisions(response["__interrupt__"][0])
        response = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,  # same thread_id as the paused run
        )

    print(f"\nAssistant: {response['messages'][-1].content}\n")


def main() -> None:
    print("Email assistant. Type 'quit' to exit.\n")
    run_turn("Hello")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("quit", "exit"):
            break
        if not user_text:
            continue
        run_turn(user_text)


if __name__ == "__main__":
    main()
