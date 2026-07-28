import os

from langchain_anthropic import ChatAnthropic

DEFAULT_MODEL = os.environ.get("KB_ANTHROPIC_MODEL", "claude-sonnet-5")


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


def _require_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    return api_key


def get_llm(model: str = DEFAULT_MODEL):
    """Non-streaming client.

    temperature is actually forwarded — it previously sat in the signature and was
    dropped, which left relevance grading and query rewriting non-reproducible.
    """
    return ChatAnthropic(
        model=model,
        api_key=_require_api_key()
    )


def get_llm_streaming(model: str = DEFAULT_MODEL):
    return ChatAnthropic(
        model=model,
        api_key=_require_api_key(),
        # temperature=temperature,
        streaming=True,
    )
