import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / ".kb" / "hybrid_index"

load_dotenv(BASE_DIR / ".env")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")

STOP_WORDS = {
    "a", "an", "and", "are", "for", "from", "how", "in",
    "is", "it", "of", "the", "to", "what",
}

DEFAULT_MIN_VECTOR_RELEVANCE = float(
    os.getenv("RAG_MIN_VECTOR_RELEVANCE", "0.35")
)
DEFAULT_MIN_BM25_SCORE = float(
    os.getenv("RAG_MIN_BM25_SCORE", "0.50")
)


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)


def slugify(text: str) -> str:
    """Convert a heading into a stable source ID component."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOP_WORDS
    ]


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            request_timeout=20,
            max_retries=1,
        )

    return _embeddings


def parse_markdown(path: Path) -> list[Section]:
    parsed: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading = path.stem.replace("_", " ").title()
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines

        content = "\n".join(current_lines).strip()
        if not content:
            current_lines = []
            return

        heading_path = [
            title
            for _, title in heading_stack
        ] or [current_heading]

        section_id = f"{path.name}#{slugify(current_heading)}"
        full_text = "\n".join([*heading_path, content])

        parsed.append(
            Section(
                id=section_id,
                file=path.name,
                heading=current_heading,
                heading_path=heading_path,
                content=content,
                tokens=tokenize(full_text),
            )
        )

        current_lines = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)

        if match:
            flush()

            level = len(match.group(1))
            current_heading = match.group(2).strip()

            heading_stack = [
                (lvl, title)
                for lvl, title in heading_stack
                if lvl < level
            ]
            heading_stack.append((level, current_heading))
        else:
            current_lines.append(line)

    flush()
    return parsed


def rebuild_bm25_stats() -> None:
    global doc_freq, avg_doc_len, files_indexed, sections_indexed

    files_indexed = len({sec.file for sec in sections})
    sections_indexed = len(sections)
    doc_freq = Counter()

    for sec in sections:
        doc_freq.update(set(sec.tokens))

    avg_doc_len = (
        sum(len(sec.tokens) for sec in sections) / len(sections)
        if sections
        else 0.0
    )


def bm25_score(
    query_tokens: list[str],
    section: Section,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not sections or not section.tokens or avg_doc_len == 0:
        return 0.0

    counts = Counter(section.tokens)
    score = 0.0

    for term in query_tokens:
        if term not in counts:
            continue

        n_docs = len(sections)
        idf = math.log(
            1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5)
        )

        tf = counts[term]
        length_norm = 1 - b + b * (len(section.tokens) / avg_doc_len)
        score += idf * ((tf * (k1 + 1)) / (tf + k1 * length_norm))

    return score


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global sections, vectorstore, files_indexed, sections_indexed

    markdown_files = sorted(docs_dir.glob("*.md"))
    new_sections: list[Section] = []
    vector_docs: list[Document] = []

    for path in markdown_files:
        parsed_sections = parse_markdown(path)
        new_sections.extend(parsed_sections)

        for sec in parsed_sections:
            vector_docs.append(
                Document(
                    page_content="\n".join([*sec.heading_path, sec.content]),
                    metadata={
                        "source": sec.id,
                        "section_id": sec.id,
                        "file": sec.file,
                        "heading": " > ".join(sec.heading_path),
                    },
                )
            )

    sections = new_sections
    rebuild_bm25_stats()

    chunks = splitter.split_documents(vector_docs)

    if chunks:
        vectorstore = FAISS.from_documents(chunks, get_embeddings())
    else:
        vectorstore = None

    files_indexed = len(markdown_files)
    sections_indexed = len(sections)

    save_hybrid_index()
    return files_indexed, sections_indexed


def save_hybrid_index(index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    if vectorstore:
        vectorstore.save_local(str(index_dir))

    bm25_payload = {
        "sections": [
            {
                "id": sec.id,
                "file": sec.file,
                "heading": sec.heading,
                "heading_path": sec.heading_path,
                "content": sec.content,
                "tokens": sec.tokens,
            }
            for sec in sections
        ],
        "stats": {
            "files_indexed": files_indexed,
            "sections_indexed": sections_indexed,
        },
    }

    (index_dir / "bm25_index.json").write_text(
        json.dumps(bm25_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_hybrid_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global sections, vectorstore, files_indexed, sections_indexed

    bm25_file = index_dir / "bm25_index.json"

    if bm25_file.exists():
        payload = json.loads(bm25_file.read_text(encoding="utf-8"))

        sections = [
            Section(
                id=item["id"],
                file=item["file"],
                heading=item["heading"],
                heading_path=item["heading_path"],
                content=item["content"],
                tokens=item["tokens"],
            )
            for item in payload.get("sections", [])
        ]

        rebuild_bm25_stats()

    if (index_dir / "index.faiss").exists():
        # Load only indexes created by this local application.
        vectorstore = FAISS.load_local(
            str(index_dir),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    return files_indexed, sections_indexed


def search_bm25(
    query: str,
    top_k: int = 10,
    min_score: float = DEFAULT_MIN_BM25_SCORE,
) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)

    ranked = [
        (sec, bm25_score(query_tokens, sec))
        for sec in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)

    return [
        (sec, score)
        for sec, score in ranked[:top_k]
        if score >= min_score and score > 0
    ]


def search_vector(
    query: str,
    top_k: int = 10,
) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []

    return vectorstore.similarity_search_with_score(query, k=top_k)


def vector_distance_to_relevance(distance: float) -> float:
    """
    Convert FAISS distance into a simple bounded relevance-like score.

    This is not a calibrated probability.
    """
    distance = max(float(distance), 0.0)
    return 1.0 / (1.0 + distance)


def search_vector_relevant(
    query: str,
    top_k: int = 10,
    min_relevance: float = DEFAULT_MIN_VECTOR_RELEVANCE,
) -> list[tuple[Document, float]]:
    raw_results = search_vector(query, top_k=top_k)
    best_by_section: dict[str, tuple[Document, float]] = {}

    for doc, distance in raw_results:
        relevance = vector_distance_to_relevance(distance)

        if relevance < min_relevance:
            continue

        section_id = (
            doc.metadata.get("section_id")
            or doc.metadata.get("source")
        )

        if not section_id:
            continue

        current = best_by_section.get(section_id)

        if current is None or relevance > current[1]:
            best_by_section[section_id] = (doc, relevance)

    return sorted(
        best_by_section.values(),
        key=lambda item: item[1],
        reverse=True,
    )


def search_hybrid(
    query: str,
    top_k: int = 3,
    rrf_k: int = 60,
) -> list[dict]:
    """Combine BM25 and vector rankings using Reciprocal Rank Fusion."""

    bm25_results = search_bm25(query, top_k=10)
    vector_results = search_vector_relevant(query, top_k=10)

    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, (sec, _) in enumerate(bm25_results, start=1):
        section_id = sec.id

        rrf_scores[section_id] = (
            rrf_scores.get(section_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

        doc_map[section_id] = {
            "id": section_id,
            "heading": " > ".join(sec.heading_path),
            "content": sec.content,
        }

    for rank, (doc, _) in enumerate(vector_results, start=1):
        section_id = (
            doc.metadata.get("section_id")
            or doc.metadata.get("source")
        )

        if not section_id:
            continue

        rrf_scores[section_id] = (
            rrf_scores.get(section_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

        if section_id not in doc_map:
            doc_map[section_id] = {
                "id": section_id,
                "heading": doc.metadata.get("heading", "unknown"),
                "content": doc.page_content,
            }

    ranked_sections = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    final_results = []

    for section_id, rrf_score in ranked_sections[:top_k]:
        info = dict(doc_map[section_id])
        info["score"] = round(rrf_score, 4)
        final_results.append(info)

    return final_results