import os

from langchain_anthropic import ChatAnthropic


def extract_text(content) -> str:
    """Return the plain text portion of a LangChain message/chunk `content` value.

    Anthropic can stream extended-thinking content blocks even when `thinking`
    was never requested on the client. langchain_anthropic represents those
    (and other non-text blocks) as list-shaped content, mixed in with plain
    str chunks for ordinary text deltas. Non-text blocks are skipped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


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
