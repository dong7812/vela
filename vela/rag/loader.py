from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_document(path: str) -> list[str]:
    file = Path(path)
    suffix = file.suffix.lower()

    if suffix in (".txt", ".md"):
        text = file.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        text = _read_pdf(file)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

    return _chunk(text)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _chunk(text: str) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return [c for c in chunks if c.strip()]
