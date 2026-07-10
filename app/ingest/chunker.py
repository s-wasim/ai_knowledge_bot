from __future__ import annotations


def chunk_file(
    path: str,
    text: str,
    window: int = 80,
    overlap: int = 20,
    min_chunk: int = 10,
) -> list[dict]:
    if not text:
        return []

    lines = text.split('\n')
    total = len(lines)

    if total <= window:
        return [{
            "path": path,
            "start_line": 1,
            "end_line": total,
            "content": text,
        }]

    step = max(1, window - overlap)
    chunks: list[dict] = []
    start = 0

    while start < total:
        end = min(start + window, total)
        chunk_lines = lines[start:end]
        chunks.append({
            "path": path,
            "start_line": start + 1,
            "end_line": end,
            "content": '\n'.join(chunk_lines),
        })
        if end == total:
            break
        start += step

    if len(chunks) > 1:
        last = chunks[-1]
        last_size = last["end_line"] - last["start_line"] + 1
        if last_size < min_chunk:
            prev = chunks[-2]
            prev["end_line"] = last["end_line"]
            prev["content"] = prev["content"] + '\n' + last["content"]
            chunks.pop()

    return chunks
