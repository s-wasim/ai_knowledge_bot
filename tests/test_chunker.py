from app.ingest.chunker import chunk_file


def test_200_line_file():
    path = "/fake/file.py"
    text = '\n'.join(f"line {i}" for i in range(1, 201))
    result = chunk_file(path, text)
    assert len(result) == 3
    assert result[0]["start_line"] == 1
    assert result[0]["end_line"] == 80
    assert result[1]["start_line"] == 61
    assert result[1]["end_line"] == 140
    assert result[2]["start_line"] == 121
    assert result[2]["end_line"] == 200


def test_30_line_file():
    path = "/fake/file.py"
    text = '\n'.join(f"line {i}" for i in range(1, 31))
    result = chunk_file(path, text)
    assert len(result) == 1
    assert result[0]["start_line"] == 1
    assert result[0]["end_line"] == 30


def test_reconstruct_coverage():
    path = "/fake/file.py"
    lines = [f"line {i}" for i in range(1, 201)]
    text = '\n'.join(lines)
    result = chunk_file(path, text)
    covered = [False] * 200
    for chunk in result:
        for i in range(chunk["start_line"] - 1, chunk["end_line"]):
            covered[i] = True
    assert all(covered)


def test_empty_file():
    assert chunk_file("/fake/file.py", "") == []


def test_exact_window():
    path = "/fake/file.py"
    text = '\n'.join(f"line {i}" for i in range(1, 81))
    result = chunk_file(path, text)
    assert len(result) == 1
    assert result[0]["start_line"] == 1
    assert result[0]["end_line"] == 80


def test_boundary_merge():
    path = "/fake/file.py"
    text = '\n'.join(f"line {i}" for i in range(1, 22))
    result = chunk_file(path, text, window=10, overlap=8, min_chunk=10)
    assert len(result) == 6
    assert result[-1]["start_line"] == 11
    assert result[-1]["end_line"] == 21


def test_overlap_correct():
    path = "/fake/file.py"
    text = '\n'.join(f"line {i}" for i in range(1, 201))
    result = chunk_file(path, text, window=80, overlap=20)
    for i in range(len(result) - 1):
        a = result[i]["content"].split('\n')
        b = result[i + 1]["content"].split('\n')
        overlap_start = result[i + 1]["start_line"] - 1
        overlap_end = result[i]["end_line"]
        a_overlap = a[-(overlap_end - overlap_start):]
        b_overlap = b[:overlap_end - overlap_start]
        assert a_overlap == b_overlap


def test_1_based_line_numbers():
    path = "/fake/file.py"
    text = '\n'.join(f"line {i}" for i in range(1, 201))
    result = chunk_file(path, text)
    assert result[0]["start_line"] == 1
