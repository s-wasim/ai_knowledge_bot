import os

from langchain_anthropic import ChatAnthropic


def get_llm(model="claude-sonnet-5", temperature=0):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    return ChatAnthropic(
        model=model,
        api_key=api_key,
    )


def get_llm_streaming(model="claude-sonnet-5", temperature=0):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    return ChatAnthropic(
        model=model,
        streaming=True,
        api_key=api_key,
    )
